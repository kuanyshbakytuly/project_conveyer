#!/bin/bash

# Video Processing Pipeline Docker Build Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="video-processor"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"
COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose is installed"
}

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime is not properly configured."
        print_error "Please install nvidia-docker2 and configure the runtime."
        print_error "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    print_status "NVIDIA Docker runtime is configured"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs Models
    
    # Set permissions
    chmod 755 data logs Models
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "$DOCKERFILE" ]; then
        print_error "Dockerfile not found!"
        exit 1
    fi
    
    # Build with buildkit for better caching
    DOCKER_BUILDKIT=1 docker build \
        --progress=plain \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f "$DOCKERFILE" \
        .
    
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Validate requirements
validate_requirements() {
    print_status "Validating project structure..."
    
    # Check for required files
    required_files=("backend.py" "processor.py" "config.py" "requirements.txt" "requirements-docker.txt")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Required file not found: $file"
        fi
    done
    
    # Check for models directory
    if [ ! -d "Models" ] || [ -z "$(ls -A Models)" ]; then
        print_warning "Models directory is empty. Make sure to add your TensorRT models."
    fi
    
    # Check for video file
    if [ ! -f "data/video.mov" ]; then
        print_warning "Video file not found at data/video.mov"
        print_warning "Make sure to place your video file in the data directory"
    fi
}

# Main build process
main() {
    echo "====================================="
    echo "Video Processing Pipeline Docker Build"
    echo "====================================="
    echo ""
    
    # Run checks
    check_docker
    check_docker_compose
    check_nvidia_docker
    
    # Create directories
    create_directories
    
    # Validate project
    validate_requirements
    
    # Build image
    build_image
    
    echo ""
    print_status "Build completed successfully!"
    echo ""
    echo "To run the application:"
    echo "  docker-compose up -d"
    echo ""
    echo "To run with monitoring stack:"
    echo "  docker-compose --profile monitoring up -d"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f video-processor"
    echo ""
    echo "Access the application at:"
    echo "  http://localhost:8000"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --tag TAG     Set image tag (default: latest)"
            echo "  --name NAME   Set image name (default: video-processor)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main build process
main