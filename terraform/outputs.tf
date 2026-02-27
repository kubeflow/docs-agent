# ---------------------------------------------------------------------------
# Root module outputs
# ---------------------------------------------------------------------------

output "cluster_endpoint" {
  description = "OKE cluster API server endpoint."
  value       = module.oke.cluster_endpoint
}

output "milvus_endpoint" {
  description = "Internal Milvus service endpoint."
  value       = "${module.milvus.service_host}:${module.milvus.service_port}"
}

output "kserve_inference_url" {
  description = "KServe InferenceService URL."
  value       = module.kserve.inference_url
}

output "docs_agent_url" {
  description = "Docs-agent service URL."
  value       = module.docs_agent.service_url
}
