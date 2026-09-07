.PHONY: help test lint build deploy clean

help:
	@echo "Available commands:"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make build       - Build Docker images"
	@echo "  make deploy      - Deploy to Kubernetes"
	@echo "  make clean       - Clean up"

test:
	pytest tests/ -v --cov=src

lint:
	black --check src/ app/ tests/
	flake8 src/ app/ tests/ --count --max-complexity=10 --max-line-length=127
	mypy src/ --ignore-missing-imports

build:
	docker build -t house-price-api:latest -f Dockerfile .
	docker build -t house-price-streamlit:latest -f Dockerfile.streamlit .

deploy:
	kubectl apply -f k8s/
	kubectl rollout status deployment/house-price-api -n ml-platform
	kubectl rollout status deployment/house-price-streamlit -n ml-platform

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "coverage.xml" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
