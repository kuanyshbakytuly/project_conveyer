# Multi-stage build for optimized image size
# Stage 1: Base image with CUDA and development tools
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    cmake \
    build-essential \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libpostproc-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libnvidia-decode-530 \
    libnvidia-encode-530 \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg with NVIDIA hardware acceleration
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

# Copy system dependencies from builder
COPY --from=builder /usr/lib/x86_64-linux-gnu/libav* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsw* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpostproc* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /usr/bin/ffprobe /usr/bin/ffprobe

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-distutils \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-docker.txt .

# Install PyTorch with CUDA 12.8 support (nightly build)
RUN pip install torch==2.8.0.dev20250408+cu128 torchvision==0.22.0.dev20250408+cu128 torchaudio==2.6.0.dev20250408+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install ONNX and related packages
RUN pip install onnx onnxslim onnxruntime-gpu

# Install TensorRT
RUN pip install nvidia-tensorrt

# Install other requirements
RUN pip install \
    ultralytics==8.3.105 \
    ffmpegcv[cuda]==0.3.16 \
    pycuda \
    pynvml \
    && pip install -r requirements-docker.txt

# Create directories for models and data
RUN mkdir -p /app/Models /app/data /app/logs

# Copy application code
COPY . /app/

# Copy and set permissions for entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chmod -R 755 /app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]