#!/bin/bash

# Build Docker images
echo "Building Docker images..."
docker build -t house-price-api:latest -f Dockerfile .
docker build -t house-price-streamlit:latest -f Dockerfile.streamlit .

# Create namespace
kubectl create namespace ml-platform --dry-run=client -o yaml | kubectl apply -f -

# Apply PVC and PV
kubectl apply -f k8s/persistent-volume.yaml

# Copy model files to PV (if using local hostPath)
mkdir -p /data/models
cp model.pickle /data/models/
cp params.pickle /data/models/

# Apply deployments and services
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/namespace.yaml

# Check status
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=house-price-api -n ml-platform --timeout=60s
kubectl wait --for=condition=ready pod -l app=house-price-streamlit -n ml-platform --timeout=60s

# Get service endpoints
echo "Services running:"
kubectl get svc -n ml-platform
kubectl get pods -n ml-platform
