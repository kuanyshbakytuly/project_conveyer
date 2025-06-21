# config.py - Central configuration for the video processing pipeline

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ProcessorConfig:
    """Configuration for video processor"""
    # RTSP Stream URLs - Configure your camera URLs here
    rtsp_urls: List[str] = field(default_factory=lambda: [
        # Example RTSP URLs - Replace with your actual camera URLs
        # HIKVision format: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
        # Dahua format: rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0
        # Generic format: rtsp://username:password@ip:port/path
        
        # You can set these via environment variables or directly here
        os.environ.get(f'RTSP_URL_{i}', f'rtsp://admin:admin@192.168.1.{100+i}:554/stream1')
        for i in range(20)
    ])
    
    # Legacy video file support (fallback)
    input_video_path: str = os.environ.get('VIDEO_PATH', '')
    
    # Output settings
    output_resolution: Tuple[int, int] = (1280, 720)
    frame_skip: int = int(os.environ.get('FRAME_SKIP', '2'))  # Process every Nth frame
    
    # RTSP specific settings
    use_no_buffer_mode: bool = os.environ.get('RTSP_NO_BUFFER', 'false').lower() == 'true'
    rtsp_transport: str = os.environ.get('RTSP_TRANSPORT', 'tcp')  # tcp or udp
    rtsp_timeout: int = int(os.environ.get('RTSP_TIMEOUT', '10'))  # seconds
    
    # Model paths
    detection_model_pattern: str = 'Models/model_det_fp16_{gpu_id}/best.engine'
    segmentation_model_pattern: str = 'Models/model_seg_fp16_{gpu_id}/best_yoloseg.engine'
    tracker_config: str = 'bytetrack_custom.yaml'
    
    # Warmup images
    detection_warmup_image: str = 'frame.jpg'
    segmentation_warmup_image: str = 'box.jpg'
    
    # Performance settings
    jpeg_quality: int = int(os.environ.get('JPEG_QUALITY', '80'))
    queue_size: int = int(os.environ.get('QUEUE_SIZE', '5'))
    
    # Detection settings
    detection_confidence: float = 0.6
    
    def __post_init__(self):
        """Load RTSP URLs from environment variables if set"""
        # Check for individual RTSP URL environment variables
        env_urls = []
        for i in range(100):  # Support up to 100 cameras
            url = os.environ.get(f'RTSP_URL_{i}')
            if url:
                env_urls.append(url)
            else:
                break
        
        # If we found URLs in environment, use them
        if env_urls:
            self.rtsp_urls = env_urls
        
        # Also check for comma-separated list
        urls_list = os.environ.get('RTSP_URLS')
        if urls_list:
            self.rtsp_urls = [url.strip() for url in urls_list.split(',')]


@dataclass
class BackendConfig:
    """Configuration for FastAPI backend"""
    # Server settings
    host: str = '0.0.0.0'
    port: int = int(os.environ.get('PORT', '8000'))
    
    # Stream settings
    num_streams: int = int(os.environ.get('NUM_STREAMS', '20'))
    target_fps: int = int(os.environ.get('TARGET_FPS', '25'))
    
    # WebSocket settings
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    ws_max_size: int = 16777216  # 16MB
    
    # Queue settings
    mp_queue_size: int = int(os.environ.get('MP_QUEUE_SIZE', '5'))
    async_queue_size: int = int(os.environ.get('ASYNC_QUEUE_SIZE', '5'))
    
    # Performance
    enable_access_log: bool = False
    log_level: str = os.environ.get('LOG_LEVEL', 'info')


@dataclass
class CameraConfig:
    """Camera-specific configuration"""
    resolutions = {
        (1280, 720): {
            'segmentation_area': [432, 0, 779, 720],
            'cm_per_pixel': 0.02997
        },
        (1920, 1080): {
            'segmentation_area': [648, 0, 1169, 1080],
            'cm_per_pixel': 0.01998
        },
        (960, 540): {
            'segmentation_area': [324, 0, 585, 539],
            'cm_per_pixel': 0.03996
        },
        (3840, 2160): {
            'segmentation_area': [1295, 1, 2338, 2154],
            'cm_per_pixel': 0.00999
        }
    }


# Create global config instances
processor_config = ProcessorConfig()
backend_config = BackendConfig()
camera_config = CameraConfig()


# Helper function to validate RTSP URLs
def validate_rtsp_urls():
    """Validate and log RTSP URLs configuration"""
    valid_urls = []
    for i, url in enumerate(processor_config.rtsp_urls):
        if url and (url.startswith('rtsp://') or url.startswith('http://') or url.startswith('https://')):
            valid_urls.append(url)
            print(f"Stream {i}: {url}")
        elif url and os.path.exists(url):  # Local file
            valid_urls.append(url)
            print(f"Stream {i}: Local file - {url}")
    
    if not valid_urls:
        print("WARNING: No valid RTSP URLs configured!")
        print("Set RTSP URLs using environment variables:")
        print("  RTSP_URL_0=rtsp://camera1...")
        print("  RTSP_URL_1=rtsp://camera2...")
        print("Or comma-separated:")
        print("  RTSP_URLS=rtsp://cam1,rtsp://cam2,...")
    
    return valid_urls


# Example configuration file content
EXAMPLE_RTSP_CONFIG = """
# Example RTSP configuration (.env file)

# Individual camera URLs (recommended)
RTSP_URL_0=rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
RTSP_URL_1=rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101
RTSP_URL_2=rtsp://admin:password@192.168.1.102:554/Streaming/Channels/101
# ... up to RTSP_URL_19 for 20 cameras

# Or use comma-separated list
# RTSP_URLS=rtsp://cam1:554/stream1,rtsp://cam2:554/stream1

# Stream settings
NUM_STREAMS=20
TARGET_FPS=25
FRAME_SKIP=2

# RTSP specific settings
RTSP_NO_BUFFER=false  # Set to true for absolute minimum latency
RTSP_TRANSPORT=tcp    # tcp or udp
RTSP_TIMEOUT=10       # Connection timeout in seconds

# Performance tuning
JPEG_QUALITY=80
QUEUE_SIZE=5
MP_QUEUE_SIZE=5
ASYNC_QUEUE_SIZE=5

# Logging
LOG_LEVEL=info
"""

# Save example configuration if requested
if os.environ.get('GENERATE_RTSP_CONFIG'):
    with open('.env.example', 'w') as f:
        f.write(EXAMPLE_RTSP_CONFIG)
    print("Generated .env.example file with RTSP configuration template")