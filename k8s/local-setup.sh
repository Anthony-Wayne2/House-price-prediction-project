#!/bin/bash

# Start minikube if not running
if ! minikube status &> /dev/null; then
    echo "Starting minikube..."
    minikube start --memory=4096 --cpus=4
fi

# Enable ingress addon
minikube addons enable ingress

# Build images in minikube's docker daemon
echo "Building images in minikube..."
eval $(minikube docker-env)
docker build -t house-price-api:latest -f Dockerfile .
docker build -t house-price-streamlit:latest -f Dockerfile.streamlit .

# Deploy to kubernetes
./k8s/build-and-deploy.sh

# Get minikube IP
MINIKUBE_IP=$(minikube ip)
echo "=========================================="
echo "ML Platform is running!"
echo "API: http://$MINIKUBE_IP/api/health"
echo "Streamlit: http://$MINIKUBE_IP/"
echo "=========================================="

# Add to /etc/hosts (requires sudo)
echo "To access via ingress, add to /etc/hosts:"
echo "$MINIKUBE_IP ml-platform.local"
