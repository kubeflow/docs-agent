terraform {
  required_version = ">= 1.0.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.26"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

locals {
  kubeconfig = pathexpand("~/.kube/config")
}

provider "kubernetes" {
  config_path = local.kubeconfig
}

provider "helm" {
  kubernetes {
    config_path = local.kubeconfig
  }
}

provider "kubectl" {
  config_path = local.kubeconfig
}
