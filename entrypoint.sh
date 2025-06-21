#!/bin/bash
set -e

# Function to check GPU availability
check_gpu() {
    echo "Checking GPU availability..."
    python -c "
import torch
import pycuda.driver as cuda
cuda.init()
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
"
}

# Function to validate models
check_models() {
    echo "Checking model files..."
    for gpu_id in 0 1; do
        if [ ! -f "/app/Models/model_det_fp16_${gpu_id}/best.engine" ]; then
            echo "WARNING: Detection model for GPU ${gpu_id} not found!"
        fi
        if [ ! -f "/app/Models/model_seg_fp16_${gpu_id}/best_yoloseg.engine" ]; then
            echo "WARNING: Segmentation model for GPU ${gpu_id} not found!"
        fi
    done
}

# Function to check video file
check_video() {
    if [ ! -f "${VIDEO_PATH:-/app/data/video.mov}" ]; then
        echo "WARNING: Video file not found at ${VIDEO_PATH:-/app/data/video.mov}"
    else
        echo "Video file found: ${VIDEO_PATH:-/app/data/video.mov}"
    fi
}

# Function to initialize warmup images if they don't exist
init_warmup_images() {
    if [ ! -f "/app/frame.jpg" ]; then
        echo "Creating default warmup frame..."
        python -c "
import numpy as np
import cv2
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.imwrite('/app/frame.jpg', frame)
"
    fi
    
    if [ ! -f "/app/box.jpg" ]; then
        echo "Creating default warmup box..."
        python -c "
import numpy as np
import cv2
box = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite('/app/box.jpg', box)
"
    fi
}

# Main startup sequence
echo "==================================="
echo "Video Processing Pipeline Starting"
echo "==================================="

# Run checks
check_gpu
check_models
check_video
init_warmup_images

# Export environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo ""
echo "Configuration:"
echo "- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "- NUM_STREAMS: ${NUM_STREAMS:-20}"
echo "- TARGET_FPS: ${TARGET_FPS:-25}"
echo "- VIDEO_PATH: ${VIDEO_PATH:-/app/data/video.mov}"
echo ""

# Start the application
echo "Starting backend server..."
exec python /app/backend.py "$@"