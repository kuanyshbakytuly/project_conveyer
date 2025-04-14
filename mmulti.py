import ffmpegcv
import cv2
import numpy as np
import threading
import time
from collections import deque
import tritonclient.http as httpclient
import psutil
import os
import pynvml
import torch.nn.functional as F
import torch
from ultralytics import YOLO

def preprocess(images):
    """Convert list of frames to model input format"""
    if not isinstance(images, list):
        images = [images]

    batch_input = []
    scales = []
    
    for image in images:
        if image.shape[0] == 3:  # CHW format
            original_h, original_w = image.shape[1], image.shape[2]
            image = image.transpose(1, 2, 0)
        else:
            original_h, original_w = image.shape[:2]
        
        scale_w = original_w / 640.0
        scale_h = original_h / 640.0
        scales.append((scale_w, scale_h))

        input_data = cv2.resize(image, (640, 640)) / 255.0
        input_data = input_data.astype(np.float32)
        input_data = input_data.transpose(2, 0, 1)
        batch_input.append(input_data)

    return np.stack(batch_input, axis=0), scales

def monitor_resources(stop_event):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_percent = process.cpu_percent()
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        print(f"CPU: {cpu_percent}% | GPU: {gpu_util}%")
        time.sleep(1)
    pynvml.nvmlShutdown()

def main():
    video_sources = ["video.mp4" for i in range(1)]  # Replace with actual sources
    num_sources = len(video_sources)
    frames_deque = [deque(maxlen=1) for _ in range(num_sources)]
    deque_locks = [threading.Lock() for _ in range(num_sources)]
    stop_event = threading.Event()

    # Initialize Triton client
    triton_client = httpclient.InferenceServerClient(url='localhost:8000')

    # Start capture threads
    capture_threads = []
    for i in range(num_sources):
        def capture_worker(source_id, video_source):
            cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(video_source, pix_fmt='nv12'), tensor_format='chw')
    
            while not stop_event.is_set():
                ret, frame_CHW_CUDA = cap.read_torch()
                if not ret:
                    break

                frame_CHW_CUDA = frame_CHW_CUDA.unsqueeze(0)
                frame_resized = F.interpolate(frame_CHW_CUDA, size=(640, 640), mode="bilinear", align_corners=False)/255
                
                with deque_locks[source_id]:
                    frames_deque[source_id].append(frame_resized)
            cap.release()

        thread = threading.Thread(target=capture_worker, args=(i, video_sources[i]))
        thread.start()
        capture_threads.append(thread)

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    monitor_thread.start()

    # Main processing loop
    fps_counter = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            current_frames = []
            all_available = True
            
            # Check all sources have frames
            for i in range(num_sources):
                with deque_locks[i]:
                    if not frames_deque[i]:
                        all_available = False
                        break
            
            if all_available:
                # Collect frames
                for i in range(num_sources):
                    with deque_locks[i]:
                        current_frames.append(frames_deque[i].popleft())
                
                # Process and infer
                #batch_input, scales = preprocess(current_frames)
                model = YOLO("http://localhost:8000/yolo", task="detect")
                det_res = model(torch.cat(current_frames, dim=0), verbose=False)
                
                # Update FPS
                fps_counter += 1
                if time.time() - start_time >= 1.0:
                    fps = fps_counter / (time.time() - start_time)
                    print(f"Processing FPS: {fps:.2f}")
                    fps_counter = 0
                    start_time = time.time()
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        for thread in capture_threads:
            thread.join()
        monitor_thread.join()
        print("All threads stopped")

if __name__ == "__main__":
    main()