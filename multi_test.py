import threading
import queue
import numpy as np
import time
import cv2
import tritonclient.http as httpclient
import ffmpegcv
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from collections import deque
from ultralytics import YOLO
import torch.nn.functional as F
import torch

# Initialize GPU monitoring
try:
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
    GPU_MONITOR_ENABLED = True
except:
    print("Could not initialize NVML for GPU monitoring")
    GPU_MONITOR_ENABLED = False

class PerformanceMonitor:
    def __init__(self, window_size=60):
        self.fps_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.gpu_history = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=30)
        self.processing_times = deque(maxlen=window_size)  # Track processing times
        self.last_time = time.time()
        
    def update_fps(self):
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.fps_history.append(fps)
            return fps
        return 0
        
    def update_cpu(self):
        cpu_percent = psutil.cpu_percent()
        self.cpu_history.append(cpu_percent)
        return cpu_percent
        
    def update_gpu(self):
        if GPU_MONITOR_ENABLED:
            try:
                gpu_util = nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                self.gpu_history.append(gpu_util)
                return gpu_util
            except:
                return 0
        return 0
        
    def add_processing_time(self, processing_time):
        self.processing_times.append(processing_time)
        
    def get_stats(self):
        stats = {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'cpu': np.mean(self.cpu_history) if self.cpu_history else 0,
            'gpu': np.mean(self.gpu_history) if self.gpu_history else 0,
            'processing_fps': 1/np.mean(self.processing_times) if self.processing_times else 0,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
        }
        return stats

def send_batch_request(frames, client):
    """Send batch of frames to Triton server for inference"""
    
    #Model_det

    model1 = YOLO("http://localhost:8000/yolo", task="detect")
    model2 = YOLO("http://localhost:8000/model_seg", task="segment")
    print(frames[0].shape)
    det_res = model1(torch.cat(frames, dim=0))
    det_res = model2(torch.cat(frames, dim=0))

    #potato_imgs = preprocess(det_res)

    
    return det_res


def video_stream_thread(video_source, frame_queue, video_id, monitor):
    """Thread for capturing video frames"""
    cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(video_source, pix_fmt='nv12'), tensor_format='chw')
    
    while True:
        ret, frame_CHW_CUDA = cap.read_torch()
        if not ret:
            break

        frame_CHW_CUDA = frame_CHW_CUDA.unsqueeze(0)
        frame_resized = F.interpolate(frame_CHW_CUDA, size=(640, 640), mode="bilinear", align_corners=False)/255
        
        # Record capture time and put in queue
        capture_time = time.time()
        frame_queue.put((video_id, frame_resized, capture_time))
        monitor.update_fps()

    cap.release()

def prediction_thread(frame_queue, total_videos, client, monitor):
    """Thread for processing frames and running inference"""
    frames_batch = {i: [] for i in range(1, total_videos + 1)}
    capture_times = {i: [] for i in range(1, total_videos + 1)}  # Track capture times
    
    while True:
        video_id, frame, capture_time = frame_queue.get()
        if frame is None:  # Termination signal
            break

        frames_batch[video_id].append(frame)
        capture_times[video_id].append(capture_time)

        # Process when we have at least one frame from each source
        if all(len(frames_batch[i]) > 0 for i in range(1, total_videos + 1)):
            batch = []
            batch_capture_times = []
            
            for vid in range(1, total_videos + 1):
                batch.append(frames_batch[vid].pop(0))
                batch_capture_times.append(capture_times[vid].pop(0))
            
            # Calculate average capture time for this batch
            avg_capture_time = np.mean(batch_capture_times)
            
            # Time the inference
            start_time = time.time()
            predictions = send_batch_request(batch, client)
            inference_time = time.time() - start_time
            
            # Calculate end-to-end processing time
            end_time = time.time()
            processing_time = end_time - avg_capture_time
            monitor.add_processing_time(processing_time)
            
            # Update monitoring
            monitor.update_gpu()
            monitor.update_cpu()
            
            # Calculate metrics
            proc_time_ms = processing_time * 1000
            batch_size = len(batch)
            processing_fps = batch_size / processing_time
            
            stats = monitor.get_stats()
            
            print(f"Batch {batch_size} | "
                  f"Processing Time: {proc_time_ms:.2f}ms | "
                  f"Processing FPS: {processing_fps:.1f} | "
                  f"System FPS: {stats['fps']:.1f} | "
                  f"GPU: {stats['gpu']:.1f}% | "
                  f"Avg Processing FPS: {stats['processing_fps']:.1f}")

def main():
    # Configuration
    video_path = 'video.mp4'
    num_streams = 1
    video_sources = [video_path] * num_streams
    
    # Initialize components
    frame_queue = queue.Queue(maxsize=8)
    triton_client = httpclient.InferenceServerClient(
        url='localhost:8000',
        verbose=False,
        connection_timeout=60.0
    )
    monitor = PerformanceMonitor()

    # Start video capture threads
    video_threads = []
    for video_id, source in enumerate(video_sources, start=1):
        t = threading.Thread(
            target=video_stream_thread,
            args=(source, frame_queue, video_id, monitor)
        )
        t.start()
        video_threads.append(t)

    # Start prediction thread
    prediction_t = threading.Thread(
        target=prediction_thread,
        args=(frame_queue, len(video_sources), triton_client, monitor)
    )
    prediction_t.start()

    # Wait for completion
    for t in video_threads:
        t.join()
    
    frame_queue.put((None, None, None))  # Signal prediction thread to exit
    prediction_t.join()

    # Print final stats
    final_stats = monitor.get_stats()
    print("\nFinal Performance Statistics:")
    print(f"Average System FPS: {final_stats['fps']:.2f}")
    print(f"Average Processing FPS: {final_stats['processing_fps']:.2f}")
    print(f"Average Processing Time: {final_stats['avg_processing_time']*1000:.2f}ms")
    print(f"Average CPU Usage: {final_stats['cpu']:.1f}%")
    print(f"Average GPU Usage: {final_stats['gpu']:.1f}%")

if __name__ == "__main__":
    main()