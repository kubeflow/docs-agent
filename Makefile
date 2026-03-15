# docs-agent Local Development Setup Makefile

.PHONY: all clean cluster load-images deploy-milvus wait-milvus port-forward test-milvus

NAMESPACE ?= docs-agent
MILVUS_RELEASE ?= my-release
MILVUS_IMAGE ?= milvusdb/milvus:v2.6.11
MINIO_IMAGE ?= minio/minio:RELEASE.2024-12-18T13-15-44Z

all: cluster load-images deploy-milvus wait-milvus
	@echo "✅ Local deployment complete!"
	@echo "Run 'make port-forward' in one terminal and 'make test-milvus' in another."

clean:
	@echo "🧹 Cleaning up Docker containers and Minikube..."
	-docker rm -f $$(docker ps -aq) 2>/dev/null || true
	-docker container prune -f
	-minikube delete

cluster:
	@echo "🚀 Starting Minikube cluster (4 CPUs, 6GB RAM)..."
	minikube start --driver=docker --cpus=4 --memory=6144mb
	@echo "📊 Opening Minikube dashboard in the background..."
	minikube dashboard &

load-images:
	@echo "📦 Pulling and loading images into Minikube to speed up deployment..."
	docker pull $(MILVUS_IMAGE)
	docker pull $(MINIO_IMAGE)
	minikube image load $(MILVUS_IMAGE)
	minikube image load $(MINIO_IMAGE)

deploy-milvus:
	@echo "⚙️ Setting up namespace and Helm repos..."
	-kubectl create namespace $(NAMESPACE)
	helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
	helm repo update
	@echo "🚀 Deploying Milvus (Standalone mode)..."
	helm upgrade --install $(MILVUS_RELEASE) zilliztech/milvus -n $(NAMESPACE) \
	  --set cluster.enabled=false \
	  --set standalone.enabled=true \
	  --set etcd.replicaCount=1 \
	  --set etcd.persistence.enabled=false \
	  --set minio.mode=standalone \
	  --set minio.replicas=1 \
	  --set pulsar.enabled=false \
	  --set pulsarv3.enabled=false \
	  --set standalone.podAnnotations."sidecar\.istio\.io/inject"="false"

wait-milvus:
	@echo "⏳ Waiting for Milvus pods to be ready..."
	kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=$(MILVUS_RELEASE) -n $(NAMESPACE) --timeout=300s
	@echo "✅ Milvus is ready!"

port-forward:
	@echo "🔌 Port forwarding Milvus to localhost:19530..."
	kubectl port-forward svc/$(MILVUS_RELEASE)-milvus -n $(NAMESPACE) 19530:19530

test-milvus:
	@echo "🧪 Testing Milvus connection..."
	pip install pymilvus
	python3 -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ MILVUS CONNECTED SUCCESSFULLY!')"
