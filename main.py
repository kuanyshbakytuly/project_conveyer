import ffmpegcv
import multiprocessing as mp
import torch
import time
import numpy as np
import os
import cv2
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import processor_config, camera_config, backend_config

# Import your custom modules
from cv_api.src.detection.detection import PotatoDetector
from cv_api.src.segmentation.segmentation import PotatoSegmentation


class StreamProcessor:
    """Handles processing of a single video stream"""
    
    def __init__(self, stream_id: int, gpu_id: int, config: dict):
        self.stream_id = stream_id
        self.gpu_id = gpu_id
        self.config = config
        self.frame_count = 0
        self.start_time = None
        self.last_reconnect_time = 0
        self.reconnect_delay = 5.0  # seconds
        
    def initialize_models(self):
        """Initialize detection and segmentation models"""
        try:
            # Initialize CUDA context
            import pycuda.autoinit
            torch.cuda.init()
            torch.cuda.set_device(self.gpu_id)
            
            # Test CUDA availability
            test_tensor = torch.randn(10, device=f'cuda:{self.gpu_id}')
            del test_tensor
            
            logger.info(f"Initializing models for stream {self.stream_id} on GPU {self.gpu_id}")
            
            # Initialize detector
            self.detector = PotatoDetector(
                model_path=processor_config.detection_model_pattern.format(gpu_id=self.gpu_id),
                camera_id=self.stream_id,
                task='detect',
                tracker_config=processor_config.tracker_config,
                segmentation_area=self.config['segmentation_area'],
                track_ids=set(),
                warmup_image=processor_config.detection_warmup_image,
                device=self.gpu_id
            )
            
            # Initialize segmenter
            self.segmenter = PotatoSegmentation(
                model_path=processor_config.segmentation_model_pattern.format(gpu_id=self.gpu_id),
                ratio=self.config['cm_per_pixel'],
                warmup_image=processor_config.segmentation_warmup_image,
                device=self.gpu_id
            )
            
            logger.info(f"Models initialized for stream {self.stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models for stream {self.stream_id}: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        """Process a single frame"""
        metrics = {}
        
        # Skip frames based on frame_skip configuration
        if frame_id % processor_config.frame_skip != 0:
            return frame, metrics
        
        with torch.no_grad():
            # Ensure frame is contiguous in memory
            frame = np.ascontiguousarray(frame)
            
            # Detection
            det_start = time.time()
            track_result = self.detector.track(frame=frame, frame_id=frame_id)
            potato_images, potato_boxes, tracker_time, processing_time, drawing_time = track_result
            
            metrics['detection_time'] = time.time() - det_start
            metrics['tracker_time'] = tracker_time
            metrics['processing_time'] = processing_time
            metrics['drawing_time'] = drawing_time
            
            # Segmentation (only if potatoes detected)
            if potato_images:
                seg_start = time.time()
                self.detector.tracked_sizes, frame = self.segmenter.process_batch(
                    potato_images,
                    potato_boxes,
                    self.detector.tracked_sizes,
                    frame
                )
                metrics['segmentation_time'] = time.time() - seg_start
            else:
                metrics['segmentation_time'] = 0
        
        return frame, metrics
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'detector'):
            del self.detector
        if hasattr(self, 'segmenter'):
            del self.segmenter
        torch.cuda.empty_cache()


def open_rtsp_stream(stream_url: str, stream_id: int, gpu_id: int, use_gpu_decode: bool = True) -> Optional[object]:
    """Open RTSP stream with appropriate settings"""
    try:
        if stream_url.startswith('rtsp://'):
            # RTSP stream - use low latency mode
            if processor_config.use_no_buffer_mode:
                # No buffer mode for absolute minimum latency
                logger.info(f"Opening RTSP stream {stream_id} with no buffer mode")
                cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, stream_url)
            else:
                # Low latency mode with small buffer
                logger.info(f"Opening RTSP stream {stream_id} with low latency mode")
                cap = ffmpegcv.VideoCaptureStreamRT(stream_url)
        elif stream_url.startswith('http://') or stream_url.startswith('https://'):
            # HTTP/HLS stream
            logger.info(f"Opening HTTP stream {stream_id}")
            cap = ffmpegcv.VideoCaptureStream(stream_url)
        else:
            # Local file or other source - use GPU decoding if available
            logger.info(f"Opening video file for stream {stream_id}")
            if use_gpu_decode:
                cap = ffmpegcv.VideoCaptureNV(
                    stream_url, 
                    pix_fmt='bgr24', 
                    resize=processor_config.output_resolution, 
                    gpu=gpu_id
                )
            else:
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        # Test if stream is opened successfully
        if hasattr(cap, 'isOpened'):
            if not cap.isOpened():
                raise Exception("Failed to open stream")
        
        # Read a test frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise Exception("Failed to read test frame")
        
        logger.info(f"Successfully opened stream {stream_id}: {stream_url}")
        return cap
        
    except Exception as e:
        logger.error(f"Failed to open stream {stream_id} ({stream_url}): {e}")
        return None


def process_stream(stream_url: str, stream_id: int, output_queue: mp.Queue, gpu_id: int):
    """Process a video stream and output frames to queue"""
    processor = None
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = -1  # Infinite retries
    
    try:
        # Get configuration for output resolution
        config = camera_config.resolutions.get(
            processor_config.output_resolution, 
            camera_config.resolutions[(1280, 720)]
        )
        
        # Initialize processor
        processor = StreamProcessor(stream_id, gpu_id, config)
        processor.initialize_models()
        
        while reconnect_attempts != max_reconnect_attempts:
            try:
                # Open stream
                cap = open_rtsp_stream(stream_url, stream_id, gpu_id)
                if cap is None:
                    raise Exception("Failed to open stream")
                
                frame_id = 0
                fps_counter = 0
                fps_start_time = time.time()
                last_frame_time = time.time()
                reconnect_attempts = 0  # Reset on successful connection
                
                logger.info(f"Stream {stream_id} started processing from {stream_url}")
                
                while True:
                    try:
                        ret, frame = cap.read()
                        current_time = time.time()
                        
                        # Check for timeout (no frames for 10 seconds)
                        if current_time - last_frame_time > 10.0:
                            logger.warning(f"Stream {stream_id} timeout - no frames for 10 seconds")
                            break
                        
                        if not ret or frame is None:
                            logger.warning(f"Stream {stream_id} read failed")
                            break
                        
                        last_frame_time = current_time
                        
                        # Resize frame if needed (for non-GPU decoded streams)
                        if frame.shape[:2] != processor_config.output_resolution[::-1]:
                            frame = cv2.resize(frame, processor_config.output_resolution)
                        
                        # Process frame
                        processed_frame, metrics = processor.process_frame(frame, frame_id)
                        
                        # Add timestamp and stream info
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(processed_frame, f"Stream {stream_id} - {timestamp}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Encode frame as JPEG for transmission
                        _, buffer = cv2.imencode('.jpg', processed_frame, 
                                                [cv2.IMWRITE_JPEG_QUALITY, processor_config.jpeg_quality])
                        
                        # Send to queue (non-blocking to prevent backpressure)
                        try:
                            output_queue.put_nowait(buffer.tobytes())
                        except:
                            # Queue full, skip frame
                            pass
                        
                        # Update counters
                        frame_id += 1
                        fps_counter += 1
                        
                        # Log FPS every second
                        if current_time - fps_start_time >= 1.0:
                            fps = fps_counter / (current_time - fps_start_time)
                            logger.info(f"Stream {stream_id}: {fps:.2f} FPS")
                            fps_counter = 0
                            fps_start_time = current_time
                            
                    except Exception as e:
                        logger.error(f"Error processing frame from stream {stream_id}: {e}")
                        break
                
            except Exception as e:
                logger.error(f"Stream {stream_id} connection error: {e}")
            
            finally:
                # Cleanup current connection
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
            
            # Reconnection logic
            reconnect_attempts += 1
            current_time = time.time()
            
            # Exponential backoff with max delay
            delay = min(processor.reconnect_delay * (2 ** min(reconnect_attempts - 1, 5)), 60)
            
            logger.info(f"Stream {stream_id} reconnecting in {delay:.1f} seconds (attempt {reconnect_attempts})")
            time.sleep(delay)
                    
    except Exception as e:
        logger.error(f"Stream {stream_id} fatal error: {e}")
    finally:
        # Final cleanup
        if cap:
            try:
                cap.release()
            except:
                pass
        if processor:
            processor.cleanup()
        logger.info(f"Stream {stream_id} processor terminated")


def main():
    """Main entry point for standalone testing"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU mode")
        num_gpus = 1
    else:
        num_gpus = torch.cuda.device_count()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Get stream URLs from configuration
    stream_urls = processor_config.rtsp_urls
    num_streams = min(len(stream_urls), backend_config.num_streams)
    
    if num_streams == 0:
        logger.error("No stream URLs configured!")
        return
    
    logger.info(f"Starting {num_streams} streams across {num_gpus} GPUs")
    
    # Create queues and processes
    queues = []
    processes = []
    
    for stream_id in range(num_streams):
        # Get stream URL
        stream_url = stream_urls[stream_id]
        
        # Distribute streams across available GPUs
        gpu_id = stream_id % num_gpus if torch.cuda.is_available() else 0
        
        # Create queue for this stream
        queue = mp.Queue(maxsize=processor_config.queue_size)
        queues.append(queue)
        
        # Create and start process
        process = mp.Process(
            target=process_stream,
            args=(stream_url, stream_id, queue, gpu_id),
            daemon=True
        )
        process.start()
        processes.append(process)
    
    logger.info("All streams started. Press Ctrl+C to stop.")
    
    try:
        # Keep main process alive
        while True:
            time.sleep(1)
            
            # Check if any process died
            for i, p in enumerate(processes):
                if not p.is_alive():
                    logger.warning(f"Stream {i} died, restarting...")
                    gpu_id = i % num_gpus if torch.cuda.is_available() else 0
                    processes[i] = mp.Process(
                        target=process_stream,
                        args=(stream_urls[i], i, queues[i], gpu_id),
                        daemon=True
                    )
                    processes[i].start()
                    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Terminate all processes
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()