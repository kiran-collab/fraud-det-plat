variable "region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "dev | staging | prod. Controls NAT redundancy, public endpoint access and apply_immediately."
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "name_prefix" {
  type    = string
  default = "fraudplat"
}

variable "kubernetes_version" {
  type    = string
  default = "1.29"
}

variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}

variable "availability_zones" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  type    = list(string)
  default = ["10.40.0.0/20", "10.40.16.0/20", "10.40.32.0/20"]
}

variable "public_subnet_cidrs" {
  type    = list(string)
  default = ["10.40.48.0/24", "10.40.49.0/24", "10.40.50.0/24"]
}

variable "data_subnet_cidrs" {
  type        = list(string)
  default     = ["10.40.64.0/22", "10.40.68.0/22", "10.40.72.0/22"]
  description = "Isolated subnets for MSK and ElastiCache - no NAT route."
}

variable "admin_cidrs" {
  type        = list(string)
  default     = []
  description = "CIDRs permitted to reach the EKS public endpoint. Empty in prod, where the endpoint is private."
}

variable "msk_instance_type" {
  type    = string
  default = "kafka.m5.large"
}

variable "msk_volume_size" {
  type        = number
  default     = 1000
  description = "GB per broker. Sized for 8 days of 5M txn/day at ~1KB/txn plus headroom."
}

variable "redis_node_type" {
  type        = string
  default     = "cache.r6g.xlarge"
  description = <<-DESC
    Memory-optimised. Sizing: ~4KB of serialised state per active card (7 days
    of events plus the bounded novelty sets). At ~2M active cards that is ~8GB,
    so r6g.xlarge (26GB) leaves room for fragmentation and burst growth without
    triggering LRU eviction in normal operation.
  DESC
}
