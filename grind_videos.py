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

if __name__ == '__main__':
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
        #out = ffmpegcv.VideoWriter(output_stream, 'h264', 25) 
        # Define frame size (e.g., 1280x720) and fps
        frame_width = 6400
        frame_height = 2880
        # fps = 25
        out = cv2.VideoWriter('outpy20.avi',
                       cv2.VideoWriter_fourcc('M','J','P','G'), 
                       25, 
                       (frame_width, frame_height))

        # # Define the codec — 'mp4v' for .mp4 or 'XVID' for .avi
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi

        # # Create the VideoWriter object
        # out = cv2.VideoWriter('output1.mp4', fourcc, fps, (frame_width, frame_height))

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

                # Tracking postprocessing timing
                start_post_track = time.time()
                annotated_frame = cv2_image.copy()
                cv2.rectangle(annotated_frame, (area_for_segment[0], area_for_segment[1]),
                               (area_for_segment[2], area_for_segment[3]), (255, 0, 0), 2)
                potato_boxes = []
                potato_img_boxes = []

                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is None:
                            continue
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        track_id = int(box.id.cpu().numpy()[0])
                        track_set.add(track_id)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        prev_size = map_track_size.get(track_id, (0, 0))
                        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"{round(prev_size[0], 2)}cm {round(prev_size[1], 2)}cm",
                                    (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                        if (area_for_segment[0] <= x1 <= area_for_segment[2] and
                            area_for_segment[1] <= y1 <= area_for_segment[3] and
                            area_for_segment[0] <= x2 <= area_for_segment[2] and
                            area_for_segment[1] <= y2 <= area_for_segment[3] and
                            prev_size[0] == 0 and prev_size[1] == 0):

                            img_box = cv2_image[y1:y2, x1:x2]
                            potato_img_boxes.append(img_box)
                            potato_boxes.append([x1, y1, x2, y2, track_id])

                        data = {
                            "camera_id": camera_id,
                            "potato_id": track_id,
                            "type": "potato",
                            "size": f"{round(prev_size[0], 2)}cm {round(prev_size[1], 2)}cm",
                            "coordinates": [x1, y1, x2, y2],
                            "frame_id": frame_count,
                            "passed": "yes",
                            "sorted": "no"
                        }
                        potato_data[track_id] = data
                end_post_track = time.time()
                phase_times['post_tracking'].append(end_post_track - start_post_track)

                # Segmentation processing
                if potato_img_boxes:
                    # Segmentation model timing
                    start_seg = time.time()
                    results_seg = model_seg.predict(potato_img_boxes, imgsz=320)
                    end_seg = time.time()
                    phase_times['model_seg'].append(end_seg - start_seg)

                    # Segmentation postprocessing timing
                    start_post_seg = time.time()
                    frame_calc_sizes = 0
                    for i, result in enumerate(results_seg):
                        x1, y1, x2, y2, track_id = potato_boxes[i]
                        prev_major, prev_minor = map_track_size.get(track_id, (0, 0))

                        if result.masks:
                            for mask in result.masks.xy:
                                abs_coords = mask + np.array([x1, y1])
                                abs_coords = abs_coords.astype(np.int32)
                                contour = np.int32([abs_coords]).reshape((-1, 1, 2))

                                if len(contour) >= 5:
                                    # Size calculation timing
                                    start_calc = time.time()
                                    ellipse = cv2.fitEllipse(contour)
                                    _, axes, _ = ellipse
                                    major_axis = max(axes) * cm_per_pixel
                                    minor_axis = min(axes) * cm_per_pixel
                                    avg_major = (major_axis + prev_major) / 2 if prev_major else major_axis
                                    avg_minor = (minor_axis + prev_minor) / 2 if prev_minor else minor_axis
                                    map_track_size[track_id] = (avg_major, avg_minor)
                                    end_calc = time.time()
                                    frame_calc_sizes += (end_calc - start_calc)

                                    cv2.fillPoly(annotated_frame, [contour], colors)
                    end_post_seg = time.time()
                    phase_times['post_seg'].append(end_post_seg - start_post_seg - frame_calc_sizes)
                    phase_times['calculating_sizes'].append(frame_calc_sizes)

                Count = 0 if len(track_set) == 0 else max(track_set)
                cv2.putText(annotated_frame, f"Potato Count: {Count}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                frame_w, frame_h = 1280, 720
                resized = cv2.resize(annotated_frame, (frame_w, frame_h))

                # Create 20 resized frames
                frames = [resized.copy() for _ in range(20)]

                # Arrange in a 4x5 grid
                rows = []
                for i in range(0, 20, 5):
                    row = np.hstack(frames[i:i+5])
                    rows.append(row)
                grid_frame = np.vstack(rows)

                # Draw grid lines
                grid_h, grid_w, _ = grid_frame.shape
                for col in range(1, 5):
                    x = col * frame_w
                    cv2.line(grid_frame, (x, 0), (x, grid_h), (0, 255, 0), 4)

                for row in range(1, 4):
                    y = row * frame_h
                    cv2.line(grid_frame, (0, y), (grid_w, y), (0, 255, 0), 4)

                # Outer border
                cv2.rectangle(grid_frame, (0, 0), (grid_w - 1, grid_h - 1), (0, 255, 0), 4)

                # Convert to correct format for VideoWriter
                grid_frame = np.clip(grid_frame, 0, 255).astype(np.uint8)

            # Video writing timing
            start_video_write = time.time()
            out.write(grid_frame.astype(np.uint8))
            end_video_write = time.time()
            phase_times['video_write'].append(end_video_write - start_video_write)

            frame_count += 1
            frame_times.append(time.time() - frame_start_time)

            # Periodic reporting
            current_time = time.time()
            if current_time - last_log_time >= metrics_window:
                window_frame_count = len(frame_times)
                if window_frame_count > 0:
                    avg_fps = window_frame_count / (current_time - last_log_time)
                    avg_frame_time = sum(frame_times) / window_frame_count
                    
                    gpu_usage = 0
                    gpu_mem = 0
                    if gpu_handle:
                        try:
                            util = nvmlDeviceGetUtilizationRates(gpu_handle)
                            mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
                            gpu_usage = util.gpu
                            gpu_mem = (mem_info.used / mem_info.total) * 100
                        except NVMLError:
                            pass
                    
                    phase_percent = {
                        phase: (sum(times)/len(times))/avg_frame_time*100 
                        for phase, times in phase_times.items() if times
                    }
                    
                    # Calculate absolute times for each phase
                    phase_abs_times = {
                        phase: sum(times)/len(times) if times else 0
                        for phase, times in phase_times.items()
                    }
                    
                    result_queue.put({
                        'stream_id': stream_id,
                        'fps': avg_fps,
                        'cpu': process.cpu_percent(),
                        'gpu': gpu_usage,
                        'gpu_mem': gpu_mem,
                        'frame_time': avg_frame_time,
                        'phase_percent': phase_percent,
                        'phase_times': phase_abs_times,
                        'timestamp': current_time
                    })
                
                frame_times.clear()
                for phase in phase_times:
                    phase_times[phase].clear()
                last_log_time = current_time

    except Exception as e:
        print(f"[Process {stream_id}] Error: {str(e)}")
        return
    finally:
        if 'cap' in locals():
            cap.release()
        if 'model_det' in locals():
            del model_det
        if 'model_seg' in locals():
            del model_seg
        if gpu_handle:
            nvmlShutdown()
        out.release()
        torch.cuda.empty_cache()
        print(f"[Process {stream_id}] Resources released")

def main():
    streams = ['video.mp4'] * 1
    output_streams = [f'Outputs_test_without_models/grind_{i}.mp4'for i, inp in enumerate(streams)]
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
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    main()