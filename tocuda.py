import ffmpegcv
from ultralytics import YOLO
import multiprocessing as mp
import torch
import torch.nn.functional as F
import time
import psutil
from pynvml import *
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

if name == '__main__':
    mp.set_start_method('spawn', force=True)

def init_gpu_monitoring():
    try:
        nvmlInit()
        return nvmlDeviceGetHandleByIndex(0)
    except NVMLError:
        return None

def warm_up_model(model, device='cuda', warmup_iters=1, imgsz=640):
    print(f"[Process {os.getpid()}] Warming up model on {device}...")
    warmup_tensor = torch.randn(1, 3, imgsz, imgsz, device=device).float()
    warmup_tensor = (warmup_tensor - warmup_tensor.min()) / (warmup_tensor.max() - warmup_tensor.min())
    for _ in range(warmup_iters):
        _ = model.predict(warmup_tensor, verbose=True)
        if hasattr(model, 'track'):
            _ = model.track(warmup_tensor, verbose=True)
    print(f"[Process {os.getpid()}] Warmup complete")
    return model

def warm_up_model1(model, device='cuda', warmup_iters=1, imgsz=640):
    print(f"[Process {os.getpid()}] Warming up model on {device}...")
    warmup_tensor = torch.randn(1, 3, imgsz, imgsz, device=device).float()
    warmup_tensor = (warmup_tensor - warmup_tensor.min()) / (warmup_tensor.max() - warmup_tensor.min())
    for _ in range(warmup_iters):
        _ = model.predict(warmup_tensor, verbose=False)
        if hasattr(model, 'track'):
            _ = model.track(warmup_tensor, verbose=False)
    print(f"[Process {os.getpid()}] Warmup complete")
    return model

def process_stream(stream, stream_id, output_stream, result_queue):
    try:
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gpu_handle = init_gpu_monitoring()
        
        model_det = YOLO('Models/model_det_test/best.engine', task='detect')
        torch.cuda.empty_cache()
        
        model_seg = YOLO('Models/model_seg_fp32/best_yoloseg.engine', task='segment')

        cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(stream, pix_fmt='nv12'), tensor_format='chw')
        out = ffmpegcv.VideoWriterNV(output_stream, 'h264', 25) 

        area_for_segment = [1295, 1, 2338, 2154]
        cm_per_pixel = 0.00999000999000999
        camera_id = stream_id
        colors = [144, 80, 70]

        frame_count = 0
        map_track_size = {}
        track_set = set()
        potato_data = {}
        
        process = psutil.Process(os.getpid())
        last_log_time = time.time()
        metrics_window = 1.0 
        frame_times = []
        phase_times = {
            'frame_read': [],
            'convert_tensor': [],
            'model_det': [],
            'post_tracking': [],
            'model_seg': [],
            'post_seg': [],
            'calculating_sizes': [],
            'video_write': []
        }
        
        while True:
            # Frame capture timing
            start_frame_read = time.time()
            ret, orig_frame = cap.read_torch()
            end_frame_read = time.time()
            if not ret:
                break
            phase_times['frame_read'].append(end_frame_read - start_frame_read)
            frame_start_time = time.time()

            with torch.no_grad():
                # Tensor conversion timing
                start_convert = time.time()
                inp = orig_frame.permute(1, 2, 0).cpu().numpy()
                cv2_image = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                end_convert = time.time()
                phase_times['convert_tensor'].append(end_convert - start_convert)

                # Detection model timing
                start_det = time.time()
                results = model_det.track(cv2_image, conf=0.6, persist=True, tracker='bytetrack_custom.yaml')
                end_det = time.time()
                phase_times['model_det'].append(end_det - start_det)
                
def main():
    streams = ['video.mp4'] * 10
    output_streams = [f'Outputs_test_without_models/{i}.mp4'for i, inp in enumerate(streams)]
    result_queue = mp.Queue()
    processes = []
    
    for stream_id, source in enumerate(streams):
        p = mp.Process(
            target=process_stream,
            args=(source, stream_id, output_streams[stream_id], result_queue)
        )
        p.start()
        processes.append(p)
    
    metrics = {
        'timestamps': [],
        'fps': [],
        'cpu': [],
        'gpu': [],
        'gpu_mem': [],
        'phase_times': []
    }
    
    try:
        while any(p.is_alive() for p in processes):
            while not result_queue.empty():
                data = result_queue.get()
                metrics['timestamps'].append(data['timestamp'])
                metrics['fps'].append(data['fps'])
                metrics['cpu'].append(data['cpu'])
                metrics['gpu'].append(data['gpu'])
                metrics['gpu_mem'].append(data['gpu_mem'])
                
                # Only append phase_times if they exist in the data
                if 'phase_times' in data:
                    metrics['phase_times'].append(data['phase_times'])
                
                print(
                    f"Stream {data['stream_id']} | "
                    f"FPS: {data['fps']:.1f} | "
                    f"CPU: {data['cpu']:.1f}% | "
                    f"GPU: {data['gpu']:.1f}% | "
                    f"Mem: {data['gpu_mem']:.1f}%"
                )
                if 'phase_times' in data:
                    print("Phase times (ms):")
                    for phase, time_val in data['phase_times'].items():
                        print(f"  {phase}: {time_val*1000:.2f}ms")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Stopping processes...")
        for p in processes:
            p.terminate()
    
    for p in processes:
        p.join()
    
    if metrics['timestamps']:
        plt.figure(figsize=(20, 15))
        base_time = min(metrics['timestamps'])
        times = [t - base_time for t in metrics['timestamps']]
        
        # Plot 1: System metrics
        plt.subplot(3, 2, 1)
        plt.plot(times, metrics['fps'])
        plt.title('Total FPS Across Streams')
        plt.ylabel('Frames/sec')
        
        plt.subplot(3, 2, 2)
        plt.plot(times, metrics['cpu'])
        plt.title('CPU Utilization')
        plt.ylabel('Percentage')
        
        plt.subplot(3, 2, 3)
        plt.plot(times, metrics['gpu'])
        plt.title('GPU Utilization')
        plt.ylabel('Percentage')
        
        plt.subplot(3, 2, 4)
        plt.plot(times, metrics['gpu_mem'])
        plt.title('GPU Memory Usage')
        plt.ylabel('Percentage')
        
        # Only plot phase times if we have data
        if metrics['phase_times']:
            # Plot 2: Phase times (absolute)
            plt.subplot(3, 2, 5)
            phase_names = list(metrics['phase_times'][0].keys())
            phase_values = {phase: [] for phase in phase_names}
            
            for entry in metrics['phase_times']:
                for phase in phase_names:
                    phase_values[phase].append(entry[phase] * 1000)  # Convert to ms
            
            for phase in phase_names:
                plt.plot(times[:len(phase_values[phase])], phase_values[phase], label=phase)
            
            plt.title('Phase Execution Times')
            plt.ylabel('Time (ms)')
            plt.xlabel('Time (s)')
            plt.legend()
            
            # Plot 3: Phase time distribution (pie chart)
            plt.subplot(3, 2, 6)
            avg_phase_times = {
                phase: sum(times) / len(times) 
                for phase, times in phase_values.items()
            }
            labels = [k for k, v in avg_phase_times.items() if v > 0]
            sizes = [v for v in avg_phase_times.values() if v > 0]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)


        plt.title('Average Phase Time Distribution')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        print("Saved performance report to performance_metrics.png")
if name == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    main()