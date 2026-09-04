#!/bin/bash

echo "Testing Kubernetes deployment..."

# Get API service IP
API_SERVICE=$(kubectl get svc house-price-api-service -n ml-platform -o jsonpath='{.spec.clusterIP}')

# Test health endpoint
echo "Testing health endpoint..."
kubectl run test-curl --image=curlimages/curl -i --rm --restart=Never -- -s http://$API_SERVICE:8080/health

# Test locations endpoint
echo "Testing locations endpoint..."
kubectl run test-curl --image=curlimages/curl -i --rm --restart=Never -- -s http://$API_SERVICE:8080/locations | head -100

# Test prediction
echo "Testing prediction endpoint..."
kubectl run test-curl --image=curlimages/curl -i --rm --restart=Never -- -s -X POST http://$API_SERVICE:8080/predict -H "Content-Type: application/json" -d '{"total_sq_feet": 1200, "bathrooms": 2, "bedrooms": 3, "location": "Whitefield"}'

echo "Test complete!"
