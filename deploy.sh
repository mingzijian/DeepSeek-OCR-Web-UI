#!/bin/bash

# DeepSeek OCR Application Deployment Script
# This script helps deploy the OCR application using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="deepseek-ocr-app"
CONTAINER_NAME="deepseek-ocr"
PORT="7860"
MODEL_DIR="./models"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed. Using docker commands instead."
        return 1
    fi
    
    return 0
}

# Function to check if NVIDIA Docker is available
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null; then
        print_success "NVIDIA Docker is available"
        return 0
    elif docker info | grep -q nvidia; then
        print_success "NVIDIA Docker runtime is available"
        return 0
    else
        print_warning "NVIDIA Docker is not available. GPU acceleration will not work."
        return 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p ${MODEL_DIR}
    mkdir -p ./data
    mkdir -p ./logs
    
    print_success "Directories created"
}

# Function to check model files
check_model() {
    if [ ! -d "${MODEL_DIR}/DeepSeek-OCR" ]; then
        print_warning "DeepSeek-OCR model not found in ${MODEL_DIR}"
        print_status "Please download the model first:"
        echo "  Option 1: git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-OCR ${MODEL_DIR}/DeepSeek-OCR"
        echo "  Option 2: Download from https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR"
        echo ""
        read -p "Do you want to continue without the model? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "DeepSeek-OCR model found"
    fi
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    if docker build -t ${APP_NAME} .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to stop existing container
stop_container() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_status "Stopping existing container..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        print_success "Existing container stopped and removed"
    fi
}

# Function to run container with docker-compose
run_with_compose() {
    print_status "Starting application with docker-compose..."
    
    if docker-compose up -d; then
        print_success "Application started successfully"
        print_status "Access the application at: http://localhost:${PORT}"
    else
        print_error "Failed to start application with docker-compose"
        exit 1
    fi
}

# Function to run container with docker
run_with_docker() {
    print_status "Starting application with docker..."
    
    GPU_FLAG=""
    if check_nvidia_docker; then
        GPU_FLAG="--gpus all"
    fi
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        ${GPU_FLAG} \
        -p ${PORT}:7860 \
        -v $(pwd)/${MODEL_DIR}:/app/models \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        -e MODEL_PATH=/app/models/DeepSeek-OCR \
        -e CUDA_VISIBLE_DEVICES=0 \
        --restart unless-stopped \
        ${APP_NAME}
    
    if [ $? -eq 0 ]; then
        print_success "Application started successfully"
        print_status "Access the application at: http://localhost:${PORT}"
    else
        print_error "Failed to start application"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_status "Showing application logs (Ctrl+C to exit)..."
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "Container ${CONTAINER_NAME} is not running"
    fi
}

# Function to show status
show_status() {
    print_status "Application Status:"
    
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_success "Container is running"
        docker ps -f name=${CONTAINER_NAME}
        echo ""
        print_status "Access the application at: http://localhost:${PORT}"
    else
        print_warning "Container is not running"
    fi
}

# Function to stop application
stop_app() {
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        print_status "Stopping application with docker-compose..."
        docker-compose down
    else
        print_status "Stopping application..."
        stop_container
    fi
    print_success "Application stopped"
}

# Main deployment function
deploy() {
    print_status "Starting DeepSeek OCR Application deployment..."
    
    check_docker
    HAS_COMPOSE=$?
    
    check_nvidia_docker
    
    create_directories
    check_model
    build_image
    stop_container
    
    if [ $HAS_COMPOSE -eq 0 ] && [ -f "docker-compose.yml" ]; then
        run_with_compose
    else
        run_with_docker
    fi
    
    print_success "Deployment completed!"
    echo ""
    print_status "Next steps:"
    echo "  1. Access the application: http://localhost:${PORT}"
    echo "  2. View logs: ./deploy.sh logs"
    echo "  3. Check status: ./deploy.sh status"
    echo "  4. Stop application: ./deploy.sh stop"
}

# Help function
show_help() {
    echo "DeepSeek OCR Application Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    Deploy the application (default)"
    echo "  build     Build Docker image only"
    echo "  start     Start the application"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      Show application logs"
    echo "  status    Show application status"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_DIR    Model directory (default: ./models)"
    echo "  PORT         Application port (default: 7860)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  MODEL_DIR=/path/to/models $0 deploy"
    echo "  PORT=8080 $0 deploy"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        check_docker
        create_directories
        build_image
        ;;
    start)
        check_docker
        HAS_COMPOSE=$?
        if [ $HAS_COMPOSE -eq 0 ] && [ -f "docker-compose.yml" ]; then
            docker-compose up -d
        else
            run_with_docker
        fi
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 2
        deploy
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac