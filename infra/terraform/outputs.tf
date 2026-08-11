output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "kafka_bootstrap_brokers" {
  value       = aws_msk_cluster.transactions.bootstrap_brokers_sasl_iam
  description = "Set as KAFKA_BOOTSTRAP_SERVERS on the feature-writer deployment."
  sensitive   = true
}

output "redis_primary_endpoint" {
  value       = aws_elasticache_replication_group.online_store.primary_endpoint_address
  description = "Set as REDIS_URL on both the scoring and streaming deployments."
  sensitive   = true
}

output "model_registry_uri" {
  value       = "s3://${aws_s3_bucket.models.bucket}/registry"
  description = "Set as FP_MODEL_DIR."
}

output "audit_bucket" {
  value = aws_s3_bucket.audit.bucket
}

output "scoring_role_arn" {
  value       = module.scoring_irsa.iam_role_arn
  description = "Annotate the fraudplat-scoring service account with this."
}
