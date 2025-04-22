import ffmpegcv
import multiprocessing as mp
import torch
import time
from pynvml import *
import matplotlib.pyplot as plt
import cv2
import tensorrt as trt
import psutil
import os
from cv_api.src.detection.detection import PotatoDetector
from cv_api.src.segmentation.segmentation import PotatoSegmentation
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def generate_plots(metrics_list):
    # Initialize data collection containers
    all_fps = []
    all_cpu = []
    all_gpu = []
    time_stats = {
        'frame_read': 0.0,
        'detection_tracker': 0.0,
        'detection_processing': 0.0,
        'detection_drawing': 0.0,
        'segmentation': 0.0,
        'segmentation_drawing': 0.0
    }

    # Collect data from all processes
    total_frames = 0
    for process_metrics in metrics_list:
        for frame in process_metrics:
            total_frames += 1
            
            # Calculate FPS
            if frame['total'] > 0:
                all_fps.append(1 / frame['total'])
            else:
                all_fps.append(0)
            
            # System metrics
            all_cpu.append(frame['cpu'])
            all_gpu.append(frame['gpu'])
            
            # Timing metrics
            for key in time_stats:
                time_stats[key] += frame.get(key, 0)

    # Calculate averages
    avg_fps = np.mean(all_fps) if all_fps else 0
    avg_cpu = np.mean(all_cpu) if all_cpu else 0
    avg_gpu = np.mean(all_gpu) if all_gpu else 0

    # Calculate time percentages
    total_time = sum(time_stats.values())
    time_percentages = {k: v/total_time*100 for k, v in time_stats.items()} if total_time > 0 else {}

    # Create summary text
    summary_text = f"""Performance Summary:
-------------------
Average FPS: {avg_fps:.2f}
Average CPU Usage: {avg_cpu:.2f}%
Average GPU Usage: {avg_gpu:.2f}%

Time Distribution (% of total processing time):
- Frame Reading: {time_percentages.get('frame_read', 0):.1f}%
- Detection (Model): {time_percentages.get('detection_tracker', 0):.1f}%
- Detection (Processing): {time_percentages.get('detection_processing', 0):.1f}%
- Detection (Drawing): {time_percentages.get('detection_drawing', 0):.1f}%
- Segmentation (Model): {time_percentages.get('segmentation', 0):.1f}%
- Segmentation (Drawing): {time_percentages.get('segmentation_drawing', 0):.1f}%

Absolute Times per Frame (ms):
- Frame Reading: {time_stats['frame_read']/total_frames*1000:.2f}
- Detection Model: {time_stats['detection_tracker']/total_frames*1000:.2f}
- Detection Processing: {time_stats['detection_processing']/total_frames*1000:.2f}
- Detection Drawing: {time_stats['detection_drawing']/total_frames*1000:.2f}
- Segmentation Model: {time_stats['segmentation']/total_frames*1000:.2f}
- Segmentation Drawing: {time_stats['segmentation_drawing']/total_frames*1000:.2f}
"""

    # Print and save summary
    print(summary_text)
    with open('performance_summary.txt', 'w') as f:
        f.write(summary_text)

    # Create plots
    plt.figure(figsize=(20, 15))
    
    # Summary text
    plt.subplot(3, 3, 1)
    plt.axis('off')
    plt.text(0, 0.5, summary_text, fontfamily='monospace', fontsize=9)

    # FPS plot
    plt.subplot(3, 3, 2)
    plt.plot(all_fps, color='blue')
    plt.title('Frame Rate Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.ylim(0, max(all_fps)*1.2 if all_fps else 30)

    # CPU/GPU plot
    plt.subplot(3, 3, 3)
    plt.plot(all_cpu, label='CPU Usage', color='green')
    plt.plot(all_gpu, label='GPU Usage', color='red')
    plt.title('System Resource Usage')
    plt.xlabel('Frame Number')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)

    # Time distribution pie chart
    plt.subplot(3, 3, 4)
    labels = [
        'Frame Read', 
        'Detection Model', 
        'Detection Processing',
        'Detection Drawing',
        'Segmentation Model',
        'Segmentation Drawing'
    ]
    sizes = [time_percentages[k] for k in time_stats]
    explode = (0.1, 0, 0, 0, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Processing Time Distribution (%)')
    plt.axis('equal')

    # Detailed timing plot (ms)
    plt.subplot(3, 3, 5)
    times_ms = [t/total_frames*1000 for t in time_stats.values()]
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, times_ms, align='center', alpha=0.7)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Time (ms)')
    plt.title('Average Time per Processing Stage')
    plt.grid(True)

    # Cumulative time plot
    plt.subplot(3, 3, 6)
    cumulative_times = np.cumsum(times_ms)
    plt.plot(cumulative_times, marker='o', color='purple')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Cumulative Time (ms)')
    plt.title('Cumulative Processing Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('detailed_performance_analysis_1potok.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Performance plots saved as 'detailed_performance_analysis.png'")

def process_stream(stream, stream_id, output_stream, metrics_list):
    if True:
        import pycuda.autoinit
        torch.cuda.init()

        test_tensor = torch.randn(10, device='cuda')
        del test_tensor

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        process = psutil.Process(os.getpid())
        frame_metrics = []

        # Initialization code
        segmentation_area = [1295, 1, 2338, 2154]
        cm_per_pixel = 0.00999000999000999
        conf = 0.6
        tracker_config = 'bytetrack_custom.yaml'

        # Initialize models
        # detection = PotatoDetector(
        #     model_path='Models/model_det_test/best.engine',
        #     camera_id=stream_id,
        #     task='detect',
        #     tracker_config=tracker_config,
        #     segmentation_area=segmentation_area,
        #     track_ids=set(),
        #     warmup_image='frame.jpg'
        # )
        
        # segmentation = PotatoSegmentation(
        #     model_path='Models/model_seg_fp32/best_yoloseg.engine',
        #     ratio=cm_per_pixel,
        #     warmup_image='box.jpg'
        # )

        cap = ffmpegcv.VideoCaptureNV(stream, pix_fmt='bgr24', resize=(1920, 1080))
        frame_id = 0

        while True:
            frame_times = {
                'frame_read': 0.0,
                'detection': 0.0,
                'detection_tracker': 0.0,
                'detection_processing': 0.0,
                'detection_drawing': 0.0,
                'segmentation': 0.0,
                'segmentation_drawing': 0.0,
                'total': 0.0,
                'cpu': 0.0,
                'gpu': 0.0
            }

            # Measure frame read
            start_time = time.time()
            ret, orig_frame = cap.read()
            frame_times['frame_read'] = time.time() - start_time

            if not ret:
                break

            # Get system metrics
            frame_times['cpu'] = process.cpu_percent()
            gpu_util = nvmlDeviceGetUtilizationRates(handle)
            frame_times['gpu'] = gpu_util.gpu

            with torch.no_grad():
                inp = orig_frame.copy()

                # Detection
                det_start = time.time()
                #track_result = detection.track(frame=inp, frame_id=frame_id, conf=conf)
                #potato_images, potato_boxes = track_result[0], track_result[1]
                #tracker_time = track_result[2]
                #processing_time = track_result[3]
                #detection_drawing_time = track_result[4]
                
                #frame_times['detection_tracker'] = tracker_time
                #frame_times['detection_processing'] = processing_time
                #frame_times['detection_drawing'] = detection_drawing_time
                frame_times['detection'] = time.time() - det_start

                # Segmentation
                seg_start = time.time()
                # if potato_images:
                #     detection.tracked_sizes, annotated_frame = segmentation.process_batch(
                #         potato_images,
                #         potato_boxes,
                #         detection.tracked_sizes,
                #         inp
                #     )
                frame_times['segmentation'] = time.time() - seg_start

            # Total frame time
            frame_times['total'] = sum([
                frame_times['frame_read'],
                frame_times['detection'],
                frame_times['segmentation'],
                frame_times['segmentation_drawing']
            ])

            frame_metrics.append(frame_times)
            frame_id += 1

        metrics_list.append(frame_metrics)

    # except Exception as e:
    #     print(f"Process {stream_id} error: {str(e)}")
    # finally:
    #     if 'cap' in locals():
    #         cap.release()
    #     nvmlShutdown()
    #     del detection
    #     del segmentation
    #     torch.cuda.empty_cache()
    #     print(f"Process {stream_id} released resources")

def main():
    streams = ['video.mp4'] * 10
    output_streams = [f'output_{i}.mp4' for i in range(len(streams))]
    
    with mp.Manager() as manager:
        metrics_list = manager.list()
        processes = []
        
        for stream_id, source in enumerate(streams):
            p = mp.Process(
                target=process_stream,
                args=(source, stream_id, output_streams[stream_id], metrics_list)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        generate_plots(list(metrics_list))

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    
    mp.set_start_method('spawn')
    main()