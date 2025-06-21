Hardware
- NVIDIA GPU(s) with CUDA 12.8+ support
- Minimum 16GB RAM (32GB recommended for 20 streams)
- Fast network connection for RTSP streams

Software
- Ubuntu 20.04+ (or compatible Linux distribution)
- Docker 20.10+
- NVIDIA Docker runtime (nvidia-docker2)
- Python 3.10+
- CUDA 12.8+
- TensorRT 8.6+

Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd video-processing-pipeline
```

2. Set Up Environment

For RTSP Cameras:
```bash
# Copy environment template
cp .env.rtsp .env

# Edit with your camera URLs
nano .env

# Example camera configuration:
# RTSP_URL_0=rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
# RTSP_URL_1=rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101
```

For Video Files:
```bash
# Place video files in data directory
mkdir -p data
cp your-video.mov data/video.mov
```

3. Build and Run with Docker

```bash
# Build Docker image
make build

# Run with RTSP cameras
make rtsp

# Or run with video files
make run

# View logs
make logs
```

4. Access the Web Interface

Open http://localhost:8000 in your browser to view all video streams.

## 🔧 Configuration

Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_STREAMS` | Number of concurrent streams | 20 |
| `TARGET_FPS` | Output frame rate | 25 |
| `FRAME_SKIP` | Process every Nth frame | 2 |
| `JPEG_QUALITY` | JPEG compression quality (1-100) | 80 |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | 0,1 |

RTSP Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RTSP_URL_0` to `RTSP_URL_19` | Camera URLs | - |
| `RTSP_NO_BUFFER` | Disable buffering for low latency | false |
| `RTSP_TRANSPORT` | Transport protocol (tcp/udp) | tcp |
| `RTSP_TIMEOUT` | Connection timeout (seconds) | 10 |

Camera URL Examples

**HIKVision:**
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0
```

**Axis:**
```
rtsp://user:password@192.168.1.102/axis-media/media.amp
```

Project Structure

```
video-processing-pipeline/
├── backend.py              # FastAPI web server
├── processor.py            # Video processing engine
├── config.py              # Configuration management
├── Dockerfile             # Docker build instructions
├── docker-compose.yml     # Docker Compose for video files
├── docker-compose.rtsp.yml # Docker Compose for RTSP
├── requirements.txt       # Python dependencies
├── Makefile              # Build and run commands
├── Models/               # TensorRT model files
│   ├── model_det_fp16_0/ # Detection models
│   └── model_seg_fp16_0/ # Segmentation models
├── data/                 # Video files
├── logs/                 # Application logs
└── cv_api/               # Custom CV modules
```

Usage Examples

Test Camera Connections
```bash
# Test all configured cameras
python test_rtsp_cameras.py

# Test specific cameras
python test_rtsp_cameras.py --cameras 0,1,2

# Test single URL
python test_rtsp_cameras.py --url rtsp://admin:pass@192.168.1.100:554/stream
```

Interactive Setup
```bash
# Run setup wizard
./setup_rtsp.sh

# Quick setup
make rtsp-setup
```

Development Mode
```bash
# Run with hot-reload and fewer streams
make dev

# Open shell in container
make shell

# View performance metrics
make perf
```

API Endpoints

- `GET /` - Web interface with live streams
- `GET /health` - Health check endpoint
- `GET /api/streams` - Stream information and status
- `WebSocket /ws/{stream_id}` - Live video stream

Architecture

### Processing Pipeline
```
RTSP Cameras → GPU Decoder → Detection Model → Segmentation Model → Web Stream
     ↓               ↓              ↓                 ↓                ↓
[Process 0]     [Process 1]    [Process N]      [MP Queue]      [WebSocket]
```


Monitoring

View Logs
```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f video-processor

# Filter by stream
docker-compose logs -f | grep "Stream 0"
```

Check Status
```bash
# System health
curl http://localhost:8000/health

# Stream information
curl http://localhost:8000/api/streams | jq

# GPU usage
docker exec video-processor nvidia-smi
```

 Performance Tuning

### For Low Latency
```env
RTSP_NO_BUFFER=true
RTSP_TRANSPORT=udp
MP_QUEUE_SIZE=2
FRAME_SKIP=1
```

### For Stability
```env
RTSP_NO_BUFFER=false
RTSP_TRANSPORT=tcp
RTSP_TIMEOUT=30
FRAME_SKIP=2
```

### For Many Cameras (20+)
```env
FRAME_SKIP=3
TARGET_FPS=15
JPEG_QUALITY=70
# Use sub-streams instead of main streams
```

Troubleshooting

### Docker Issues

**"Cannot connect to Docker daemon"**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

**"NVIDIA runtime not found"**
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Camera Connection Issues

**"Cannot open RTSP stream"**
- Verify camera IP and credentials
- Test with VLC: `vlc rtsp://user:pass@ip:554/stream`
- Check firewall settings
- Try TCP transport: `RTSP_TRANSPORT=tcp`

**"High latency"**
- Enable no-buffer mode: `RTSP_NO_BUFFER=true`
- Use camera sub-streams
- Reduce processing load: increase `FRAME_SKIP`

### GPU Issues

**"CUDA out of memory"**
- Reduce number of streams
- Lower resolution: use 720p instead of 1080p
- Distribute across more GPUs

Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run without Docker
python backend.py

# Run tests
python test_rtsp_cameras.py
```

### Building from Source
```bash
# Build Docker image
docker build -t video-processor:latest .

# Run with custom settings
docker run --gpus all \
  --network host \
  -v $(pwd)/Models:/app/Models \
  -v $(pwd)/data:/app/data \
  video-processor:latest
```

### Adding New Features

1. Modify `processor.py` for processing logic
2. Update `backend.py` for API changes
3. Edit `config.py` for new settings
4. Update Docker files if dependencies change

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- YOLOv8 for object detection
- TensorRT for GPU optimization
- FFmpegCV for video processing
- FastAPI for web framework
