# kagent Setup (End-to-End)

This repo’s **end-to-end deployment** lives in `kagent-feast-mcp/`.

It includes:
- `kagent-feast-mcp/manifests/`: Milvus/Istio policies, MCP server, and kagent setup
- `kagent-feast-mcp/feast_repo/`: Feast feature repo for Milvus online store
- `kagent-feast-mcp/mcp-server/`: MCP server implementation + Dockerfile
- `kagent-feast-mcp/pipelines/kubeflow-pipeline.py`: KFP pipeline (compile + upload) that works independently

## Prerequisites

- Kubernetes cluster + `kubectl`
- Helm 3.x
- A namespace you can deploy into (use `<YOUR_NAMESPACE>`)
- Groq API key (or any OpenAI-compatible provider used by kagent)
- Container registry access (to push the MCP server image)

## 0) Set placeholders

Update these placeholders before deploying:
- `<YOUR_NAMESPACE>` in all manifests
- `<YOUR_DOCKERHUB_USERNAME>` in `kagent-feast-mcp/manifests/mcp-server/mcp-server.yaml`
- `<YOUR_GROQ_API_KEY>` in `kagent-feast-mcp/manifests/kagent/setup.yaml`

## 1) Install Milvus (Helm)

```bash
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
helm repo update

helm upgrade --install milvus zilliztech/milvus -n <YOUR_NAMESPACE> \
  --set cluster.enabled=false \
  --set standalone.enabled=true \
  --set etcd.replicaCount=1 \
  --set etcd.persistence.enabled=false \
  --set minio.mode=standalone \
  --set minio.replicas=1 \
  --set pulsar.enabled=false \
  --set pulsarv3.enabled=false
```

```bash
kubectl get pods -n <YOUR_NAMESPACE> -l app.kubernetes.io/instance=milvus
```

## 2) (If using Istio deny-all) allow Milvus traffic

Apply the Istio AuthorizationPolicies:

```bash
kubectl apply -f kagent-feast-mcp/manifests/istio/
```

## 3) Deploy Feast feature repo (local apply)

```bash
cd kagent-feast-mcp/feast_repo
pip install 'feast[milvus]'
feast apply
```

## 4) Deploy the MCP server

1) Build + push your image from `kagent-feast-mcp/mcp-server/` (update the image name in the manifest).

2) Apply the Kubernetes manifest:

```bash
kubectl apply -f kagent-feast-mcp/manifests/mcp-server/mcp-server.yaml
kubectl get pods -n <YOUR_NAMESPACE> -l app=mcp-kubeflow-docs
```

## 5) Install kagent + register the agent

Install kagent:

```bash
helm install kagent-crds oci://ghcr.io/kagent-dev/kagent/helm/kagent-crds --namespace <YOUR_NAMESPACE>

helm install kagent oci://ghcr.io/kagent-dev/kagent/helm/kagent \
  --namespace <YOUR_NAMESPACE> \
  --set agents.argo-rollouts-agent.enabled=false \
  --set agents.cilium-debug-agent.enabled=false \
  --set agents.cilium-manager-agent.enabled=false \
  --set agents.cilium-policy-agent.enabled=false \
  --set agents.helm-agent.enabled=false \
  --set agents.istio-agent.enabled=false \
  --set agents.k8s-agent.enabled=false \
  --set agents.kgateway-agent.enabled=false \
  --set agents.observability-agent.enabled=false \
  --set agents.promql-agent.enabled=false \
  --set tools.grafana-mcp.enabled=false \
  --set tools.querydoc.enabled=false
```

Register the Groq secret, model config, RemoteMCPServer, and Agent:

```bash
kubectl apply -f kagent-feast-mcp/manifests/kagent/setup.yaml
kubectl get agents,remotemcpservers,modelconfigs -n <YOUR_NAMESPACE>
```

## 6) Run the indexing pipeline (compile + upload)

This pipeline is self-contained in `kagent-feast-mcp/pipelines/kubeflow-pipeline.py`.

```bash
cd kagent-feast-mcp/pipelines
pip install kfp
python kubeflow-pipeline.py
```

Upload the generated `github_rag_pipeline.yaml` to the Kubeflow Pipelines UI and run it.

## 7) Open the kagent UI

```bash
kubectl -n <YOUR_NAMESPACE> port-forward service/kagent-ui 8080:8080
```

Then open `http://localhost:8080` and chat with `kubeflow-docs-agent`.

## Quick debug

```bash
kubectl get pods -n <YOUR_NAMESPACE>
kubectl logs -f deployment/mcp-kubeflow-docs -n <YOUR_NAMESPACE>
kubectl get agents,remotemcpservers,modelconfigs -n <YOUR_NAMESPACE>
```
