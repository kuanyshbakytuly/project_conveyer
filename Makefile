# Makefile for Video Processing Pipeline

.PHONY: help build run stop logs clean dev prod test

# Default target
help:
	@echo "Video Processing Pipeline - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       Build Docker image"
	@echo "  run         Run application (production mode)"
	@echo "  dev         Run application (development mode)"
	@echo "  stop        Stop all containers"
	@echo "  logs        View application logs"
	@echo "  clean       Clean up containers and images"
	@echo "  test        Test GPU and dependencies"
	@echo "  shell       Open shell in container"
	@echo "  monitor     Run with monitoring stack"
	@echo ""

# Build Docker image
build:
	@echo "Building Docker image..."
	@./build.sh

# Run in production mode
run: build
	@echo "Starting application in production mode..."
	@docker-compose up -d
	@echo "Application started! Access at http://localhost:8000"

# Run in development mode
dev:
	@echo "Starting application in development mode..."
	@docker-compose -f docker-compose.dev.yml up

# Stop all containers
stop:
	@echo "Stopping all containers..."
	@docker-compose down
	@docker-compose -f docker-compose.dev.yml down

# View logs
logs:
	@docker-compose logs -f video-processor

# Clean up
clean: stop
	@echo "Cleaning up..."
	@docker system prune -f
	@docker volume prune -f
	@rm -rf logs/*.log

# Test GPU availability
test:
	@echo "Testing GPU availability..."
	@docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Open shell in container
shell:
	@docker-compose exec video-processor /bin/bash

# Run with monitoring
monitor: build
	@echo "Starting with monitoring stack..."
	@docker-compose --profile monitoring up -d
	@echo "Services started:"
	@echo "  - Application: http://localhost:8000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

# Check system status
status:
	@echo "System Status:"
	@echo "--------------"
	@docker-compose ps
	@echo ""
	@echo "GPU Status:"
	@echo "-----------"
	@docker-compose exec video-processor nvidia-smi || echo "Container not running"
	@echo ""
	@echo "Health Check:"
	@echo "-------------"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not available"

# Quick restart
restart: stop run

# Build without cache
rebuild:
	@echo "Rebuilding without cache..."
	@docker-compose build --no-cache

# Tail logs with timestamp
logs-tail:
	@docker-compose logs -f --tail=100 --timestamps video-processor

# Performance monitoring
perf:
	@echo "Starting performance monitoring..."
	@watch -n 1 'docker stats --no-stream video-processor && echo "" && docker-compose exec video-processor nvidia-smi'