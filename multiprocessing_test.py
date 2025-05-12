import ffmpegcv
import multiprocessing as mp
import torch
import time
from pynvml import *
import matplotlib.pyplot as plt
import tensorrt as trt
import psutil
import os
from cv_api.src.detection.detection import PotatoDetector
from cv_api.src.segmentation.segmentation import PotatoSegmentation
import numpy as np
import glob

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def generate_plots(metrics_list, stream_fps_list):
    all_stream_fps = stream_fps_list
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

    total_frames = 0
    for process_metrics in metrics_list:
        for frame in process_metrics:
            total_frames += 1
            
            if frame['total'] > 0:
                all_fps.append(1 / frame['total'])
            else:
                all_fps.append(0)
            
            all_cpu.append(frame['cpu'])
            all_gpu.append(frame['gpu'])
            
            for key in time_stats:
                time_stats[key] += frame.get(key, 0)

    avg_fps = np.mean(all_fps) if all_fps else 0
    avg_cpu = np.mean(all_cpu) if all_cpu else 0
    avg_gpu = np.mean(all_gpu) if all_gpu else 0
    avg_stream_fps = np.mean(all_stream_fps) if all_stream_fps else 0

    total_time = sum(time_stats.values())
    time_percentages = {k: v/total_time*100 for k, v in time_stats.items()} if total_time > 0 else {}

    stream_fps_text = "\n".join([f"Stream {i}: {fps:.2f} FPS" for i, fps in enumerate(all_stream_fps)])
    
    summary_text = f"""Performance Summary:
        -------------------
        Average End-to-End Stream FPS: {avg_stream_fps:.2f}
        Individual Stream FPS:
        {stream_fps_text}

        Average Per-Frame FPS: {avg_fps:.2f}
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

    # Update plot generation to include stream FPS
    plt.figure(figsize=(20, 15))
    
    # Add stream FPS plot
    plt.subplot(3, 3, 7)
    plt.bar(range(len(all_stream_fps)), all_stream_fps, color='cyan')
    plt.title('End-to-End FPS per Stream')
    plt.xlabel('Stream ID')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.xticks(range(len(all_stream_fps)))
    
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
    plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Performance plots saved as 'detailed_performance_analysis.png'")

def process_stream(stream, stream_id, output_stream, metrics_list, stream_fps_list):
    try:
        print(f'Staring proces --- {stream_id}')
        import pycuda.autoinit
        torch.cuda.init()

        test_tensor = torch.randn(10, device='cuda')
        del test_tensor

        frame_metrics = []

        # Initialization code
        # Original values for 3840×2160:
        # segmentation_area = [1295, 1, 2338, 2154]  # [x1, y1, x2, y2]
        # cm_per_pixel = 0.00999000999000999

        # Scaled values for 1920×1080:
        # segmentation_area = [648, 0, 1169, 1080]
        # cm_per_pixel = 0.01998

        # segmentation_area = [432, 0, 779, 720]  # [x1, y1, x2, y2]
        # cm_per_pixel = 0.02997

        # Scaled values for 960×540:
        segmentation_area = [324, 0, 585, 539]  # [x1, y1, x2, y2]
        cm_per_pixel = 0.03996003996003996

        conf = 0.6
        tracker_config = 'bytetrack_custom.yaml'
        path, gpu_id = stream


        # Initialize models
        print(f'Initting Detector --- process {stream_id} ')
        detection = PotatoDetector(
            model_path=f'Models/model_det_fp16_{gpu_id}/best.engine',
            camera_id=stream_id,
            task='detect',
            tracker_config=tracker_config,
            segmentation_area=segmentation_area,
            track_ids=set(),
            warmup_image='frame.jpg',
            device=gpu_id
        )
        print(f'Initting Segmenter ---process {stream_id} ')
        segmentation = PotatoSegmentation(
            model_path=f'Models/model_seg_fp16_{gpu_id}/best_yoloseg.engine',
            ratio=cm_per_pixel,
            warmup_image='box.jpg',
            device=gpu_id
        )
        print(f'Initting VideoCapture --- process {stream_id} ')
        cap = ffmpegcv.VideoCaptureNV(path, pix_fmt='bgr24', resize=(960, 540), gpu=gpu_id)
        #out = ffmpegcv.VideoWriterNV(output_stream, 'h264', 25)

        print(f'Warm Up models with first frame --- process {stream_id} ')


        frame_id = 0
        start_time = time.time()
        print(f'Start processing -- process {stream_id}')
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

            s = time.time()
            ret, frame = cap.read()
            frame_times['frame_read'] = time.time() - s

            if not ret:
                break

            # frame_times['cpu'] = process.cpu_percent()
            # gpu_util = nvmlDeviceGetUtilizationRates(handle)
            # frame_times['gpu'] = gpu_util.gpu

            with torch.no_grad():
                frame=frame.copy()

                # Detection
                det_start = time.time()
                track_result = detection.track(frame=frame, frame_id=frame_id)
                potato_images, potato_boxes = track_result[0], track_result[1]
                tracker_time = track_result[2]
                processing_time = track_result[3]
                detection_drawing_time = track_result[4]
                
                frame_times['detection_tracker'] = tracker_time
                frame_times['detection_processing'] = processing_time
                frame_times['detection_drawing'] = detection_drawing_time
                frame_times['detection'] = time.time() - det_start

                # Segmentation
                seg_start = time.time()
                if potato_images:
                    detection.tracked_sizes, frame = segmentation.process_batch(
                        potato_images,
                        potato_boxes,
                        detection.tracked_sizes,
                        frame
                    )
                frame_times['segmentation'] = time.time() - seg_start

                # #out.write(frame)

            frame_times['total'] = sum([
                frame_times['frame_read'],
                frame_times['detection'],
                frame_times['segmentation'],
                frame_times['segmentation_drawing']
            ])

            frame_metrics.append(frame_times)

            if frame_id == 0:
                ignore_first_it = time.time()-start_time
            frame_id += 1


        end_time = time.time()
        total_time = end_time - start_time - ignore_first_it
        stream_fps = frame_id / total_time if total_time > 0 else 0
        print(f'TOTAL TIME: {round(total_time, 2)} FPS: {round(stream_fps, 2)}')
        stream_fps_list.append(stream_fps)
        metrics_list.append(frame_metrics)

    except Exception as e:
        print(f"Process {stream_id} error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        # if 'out' in locals():
        #     out.release()
        if 'detection' in locals():
            del detection
        if 'segmentation' in locals():
            del segmentation
        print(f"Process {stream_id} released resources")

def main():
    videos = glob.glob('Videos/*')
    streams = [(videos[i], i%2) for i in range(16)]
    output_streams = [f'output_{i}.mp4' for i in range(len(streams))]
    
    with mp.Manager() as manager:
        metrics_list = manager.list()
        stream_fps_list = manager.list()
        processes = []
        
        for stream_id, source in enumerate(streams):
            p = mp.Process(
                target=process_stream,
                args=(source, stream_id, output_streams[stream_id], metrics_list, stream_fps_list)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        generate_plots(list(metrics_list), stream_fps_list)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    
    mp.set_start_method('spawn')
    main()