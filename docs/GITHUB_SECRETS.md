# GitHub Secrets Configuration

## Required Secrets for CI/CD

| Secret Name | Description | Where to Get |
|-------------|-------------|--------------|
| `KUBECONFIG` | Base64 encoded kubeconfig for production | `cat ~/.kube/config \| base64` |
| `KUBECONFIG_DEV` | Base64 encoded kubeconfig for development | `cat ~/.kube/config-dev \| base64` |
| `GITHUB_TOKEN` | Automatically provided by GitHub | Auto-generated |
| `DOCKER_USERNAME` | Docker Hub username | Docker Hub account |
| `DOCKER_PASSWORD` | Docker Hub password/token | Docker Hub account |

## How to Set Up Secrets

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret with its value

## How to Generate KUBECONFIG

```bash
# For Minikube
cat ~/.kube/config | base64

# For EKS
aws eks update-kubeconfig --name ml-platform-cluster --region us-east-1
cat ~/.kube/config | base64
# Complete the ci-cd.yml file
cat > .github/workflows/ci-cd.yml << 'EOF'
name: ML Platform CI/CD Pipeline

on:
  push:
    branches: [ main, develop, feature/* ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  KUBE_NAMESPACE: ml-platform

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8 black
      - name: Lint code
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
          black --check src/
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

  build:
    name: Build and Push
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push API
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/api:latest
            ghcr.io/${{ github.repository }}/api:${{ github.sha }}
      - name: Build and push Streamlit
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.streamlit
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/streamlit:latest
            ghcr.io/${{ github.repository }}/streamlit:${{ github.sha }}

  deploy:
    name: Deploy to Kubernetes
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-kubectl@v4
        with:
          version: 'latest'
      - name: Set up kubeconfig
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBECONFIG }}" | base64 --decode > $HOME/.kube/config
      - name: Deploy
        run: |
          kubectl apply -f k8s/namespace.yaml
          kubectl apply -f k8s/persistent-volume.yaml
          kubectl apply -f k8s/api-deployment.yaml
          kubectl apply -f k8s/streamlit-deployment.yaml
          kubectl apply -f k8s/prometheus.yaml
          kubectl apply -f k8s/grafana.yaml
          kubectl apply -f k8s/alertmanager.yaml
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/house-price-api -n ${{ env.KUBE_NAMESPACE }} --timeout=5m
          kubectl rollout status deployment/house-price-streamlit -n ${{ env.KUBE_NAMESPACE }} --timeout=5m
      - name: Smoke test
        run: |
          kubectl port-forward -n ${{ env.KUBE_NAMESPACE }} service/house-price-api-service 8080:8080 &
          sleep 5
          curl -s http://localhost:8080/health | grep healthy || exit 1
          pkill -f "kubectl port-forward" || true
