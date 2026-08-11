###############################################################################
# Fraud detection platform - AWS infrastructure
#
# Scope: EKS for compute, MSK for the transaction stream, ElastiCache for the
# online feature store, S3 for the model registry and feature lake.
#
# Two decisions worth reading before changing anything:
#
#   * The online store is a single-AZ-primary ElastiCache cluster with a replica
#     in a second AZ. Multi-AZ *writes* would double the write latency on the
#     feature path, and the data is reconstructible from Kafka - so the right
#     trade is fast writes plus fast failover, not synchronous durability.
#
#   * The scoring node group is compute-optimised and does not autoscale to
#     zero. Model load plus ONNX session construction is a multi-second cold
#     start, which is fine for a scale-out but not acceptable as the first
#     request after an idle period.
###############################################################################

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws        = { source = "hashicorp/aws", version = "~> 5.40" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.27" }
  }
  backend "s3" {
    bucket         = "fraudplat-tfstate"
    key            = "platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fraudplat-tflock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project     = "fraud-detection-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
      DataClass   = "confidential" # cardholder-adjacent; drives SCP enforcement
    }
  }
}

###############################################################################
# Networking
###############################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "${var.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  # Data subnets are isolated with no NAT route: MSK and ElastiCache have no
  # reason to reach the internet, and removing the route removes an
  # exfiltration path from anything that lands in them.
  intra_subnets = var.data_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "prod"
  enable_dns_hostnames = true

  # VPC flow logs are an audit requirement for a cardholder-data environment.
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

###############################################################################
# EKS
###############################################################################

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = "${var.name_prefix}-eks"
  cluster_version = var.kubernetes_version

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access       = var.environment != "prod"
  cluster_endpoint_private_access      = true
  cluster_endpoint_public_access_cidrs = var.admin_cidrs

  # Secrets encryption with a customer-managed key, and full control-plane
  # logging - both are examination findings if absent.
  cluster_encryption_config = { resources = ["secrets"] }
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  enable_irsa = true

  eks_managed_node_groups = {
    # Latency-critical. Compute-optimised, generously sized, never scaled to
    # zero - see the header note on cold starts.
    scoring = {
      instance_types = ["c6i.2xlarge"]
      min_size       = 6
      max_size       = 60
      desired_size   = 6
      capacity_type  = "ON_DEMAND"
      labels         = { workload = "scoring" }
      taints = [{
        key    = "workload"
        value  = "scoring"
        effect = "NO_SCHEDULE" # keep batch jobs off the authorization path
      }]
    }

    # Throughput-oriented, tolerant of interruption: a rebalance costs a few
    # seconds of consumer lag, which the stream absorbs.
    streaming = {
      instance_types = ["m6i.xlarge"]
      min_size       = 4
      max_size       = 24
      desired_size   = 6
      capacity_type  = "ON_DEMAND"
      labels         = { workload = "streaming" }
    }

    # Training and batch. Spot is appropriate here: Kubeflow retries a lost
    # step, and the cost difference at 5M rows/day of feature replay is large.
    training = {
      instance_types = ["m6i.4xlarge", "m6a.4xlarge", "m5.4xlarge"]
      min_size       = 0
      max_size       = 10
      desired_size   = 0
      capacity_type  = "SPOT"
      labels         = { workload = "training" }
      taints = [{ key = "workload", value = "training", effect = "NO_SCHEDULE" }]
    }
  }
}

###############################################################################
# MSK - transaction stream
###############################################################################

resource "aws_msk_configuration" "transactions" {
  name           = "${var.name_prefix}-msk-config"
  kafka_versions = ["3.6.0"]

  server_properties = <<-PROPERTIES
    auto.create.topics.enable=false
    default.replication.factor=3
    min.insync.replicas=2
    num.partitions=24
    # 8 days: one day longer than the longest feature window (7d), so the
    # online store can be fully rebuilt by replaying the topic after a total
    # cache loss without any gap in card history.
    log.retention.hours=192
    compression.type=lz4
    unclean.leader.election.enable=false
  PROPERTIES
}

resource "aws_msk_cluster" "transactions" {
  cluster_name           = "${var.name_prefix}-msk"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = length(var.availability_zones)

  broker_node_group_info {
    instance_type   = var.msk_instance_type
    client_subnets  = module.vpc.intra_subnets
    security_groups = [aws_security_group.msk.id]
    storage_info {
      ebs_storage_info { volume_size = var.msk_volume_size }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.transactions.arn
    revision = aws_msk_configuration.transactions.latest_revision
  }

  encryption_info {
    encryption_at_rest_kms_key_arn = aws_kms_key.platform.arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  client_authentication {
    sasl { iam = true } # no long-lived broker passwords to rotate
  }

  open_monitoring {
    prometheus {
      jmx_exporter { enabled_in_broker = true }
      node_exporter { enabled_in_broker = true }
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }
}

###############################################################################
# ElastiCache - online feature store
###############################################################################

resource "aws_elasticache_replication_group" "online_store" {
  replication_group_id = "${var.name_prefix}-online-store"
  description          = "Online feature store: per-card rolling windows"

  engine         = "redis"
  engine_version = "7.1"
  node_type      = var.redis_node_type
  port           = 6379

  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled           = true

  subnet_group_name  = aws_elasticache_subnet_group.online_store.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                 = aws_kms_key.platform.arn

  # allkeys-lru, not noeviction: under memory pressure the correct behaviour is
  # to drop the coldest cards' state (which degrades one card's velocity
  # features until it is rebuilt from the stream) rather than to start
  # rejecting writes, which would stall the feature writer entirely.
  parameter_group_name = aws_elasticache_parameter_group.online_store.name

  snapshot_retention_limit = 1
  maintenance_window       = "sun:05:00-sun:06:00"
  apply_immediately        = var.environment != "prod"
}

resource "aws_elasticache_parameter_group" "online_store" {
  name   = "${var.name_prefix}-online-store-params"
  family = "redis7"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
}

resource "aws_elasticache_subnet_group" "online_store" {
  name       = "${var.name_prefix}-online-store"
  subnet_ids = module.vpc.intra_subnets
}

###############################################################################
# S3 - model registry, feature lake, audit log
###############################################################################

resource "aws_s3_bucket" "models" {
  bucket = "${var.name_prefix}-models"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  # Versioning is the rollback mechanism: a promoted model that turns out to be
  # bad is reverted by repointing `current`, and the prior artifacts must still
  # exist for that to work.
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.platform.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "audit" {
  bucket              = "${var.name_prefix}-audit"
  object_lock_enabled = true # decision records must be tamper-evident
}

resource "aws_s3_bucket_object_lock_configuration" "audit" {
  bucket = aws_s3_bucket.audit.id
  rule {
    default_retention {
      mode = "COMPLIANCE"
      # Seven years: the outer bound of chargeback dispute and regulatory
      # examination windows for card decisioning records.
      years = 7
    }
  }
}

resource "aws_s3_bucket" "features" {
  bucket = "${var.name_prefix}-features"
}

resource "aws_s3_bucket_lifecycle_configuration" "features" {
  bucket = aws_s3_bucket.features.id
  rule {
    id     = "tier-old-features"
    status = "Enabled"
    transition {
      days          = 90
      storage_class = "INTELLIGENT_TIERING"
    }
    # Training uses a 180-day window; a year of retention covers a full retrain
    # plus a backfill without keeping data indefinitely.
    expiration { days = 365 }
  }
}

###############################################################################
# KMS
###############################################################################

resource "aws_kms_key" "platform" {
  description             = "Fraud platform encryption key"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "platform" {
  name          = "alias/${var.name_prefix}"
  target_key_id = aws_kms_key.platform.key_id
}

###############################################################################
# Security groups
###############################################################################

resource "aws_security_group" "msk" {
  name   = "${var.name_prefix}-msk"
  vpc_id = module.vpc.vpc_id

  ingress {
    description = "Kafka TLS from the cluster only"
    from_port   = 9098
    to_port     = 9098
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
  # No egress rule: brokers have no reason to originate outbound traffic.
}

resource "aws_security_group" "redis" {
  name   = "${var.name_prefix}-redis"
  vpc_id = module.vpc.vpc_id

  ingress {
    description = "Redis from the cluster only"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.name_prefix}"
  retention_in_days = 90
  kms_key_id        = aws_kms_key.platform.arn
}

###############################################################################
# IRSA - pod-level AWS permissions, no node-wide credentials
###############################################################################

module "scoring_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "${var.name_prefix}-scoring"
  role_policy_arns = {
    models = aws_iam_policy.model_read.arn
    audit  = aws_iam_policy.audit_write.arn
  }
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["fraud:fraudplat-scoring"]
    }
  }
}

resource "aws_iam_policy" "model_read" {
  name        = "${var.name_prefix}-model-read"
  description = "Read-only access to the model registry"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      # Read-only by design: the scoring service must never be able to modify
      # the model it serves, so a compromised pod cannot promote a model.
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
    }]
  })
}

resource "aws_iam_policy" "audit_write" {
  name        = "${var.name_prefix}-audit-write"
  description = "Append-only access to the decision audit log"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      # PutObject without Delete: object-lock enforces retention, and the
      # absence of a delete permission means the service cannot even attempt it.
      Effect   = "Allow"
      Action   = ["s3:PutObject"]
      Resource = "${aws_s3_bucket.audit.arn}/*"
    }]
  })
}
