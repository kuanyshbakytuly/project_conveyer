import ffmpegcv
import multiprocessing as mp
import threading
import torch
import time
from pynvml import *
import matplotlib.pyplot as plt
import tensorrt as trt
import psutil
import os
# Assuming these imports exist and are correct
# from cv_api.src.detection.detection import PotatoDetector
# from cv_api.src.segmentation.segmentation import PotatoSegmentation
import numpy as np
import math
import runpy # Added for the traceback reference, not needed functionally

# --- Mock classes if real ones aren't available for testing ---
class PotatoDetector:
    def __init__(self, *args, **kwargs):
        print(f"Mock PotatoDetector initialized with args: {args}, kwargs: {kwargs}")
        self.tracked_sizes = {} # Example attribute
    def track(self, frame, frame_id):
        # Mock detection logic
        time.sleep(0.005) # Simulate work
        # Return dummy values matching original code's expectation if uncommented
        # return [], [], 0.002, 0.001, 0.002 # potato_images, potato_boxes, tracker_time, processing_time, drawing_time
        return [], [], 0, 0, 0 # Match current commented-out code structure

class PotatoSegmentation:
    def __init__(self, *args, **kwargs):
        print(f"Mock PotatoSegmentation initialized with args: {args}, kwargs: {kwargs}")
    def process_batch(self, potato_images, potato_boxes, tracked_sizes, frame):
        # Mock segmentation logic
        time.sleep(0.008) # Simulate work
        return tracked_sizes, frame # Return expected values


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
# trt.init_libnvinfer_plugins(TRT_LOGGER, "") # Potentially causes issues if called globally before spawn

# Global list to hold stream sources
ALL_STREAMS = ['video.mp4'] * 20 # Make sure video.mp4 exists or use a valid path

# --- generate_plots function ---
def generate_plots(metrics_list, stream_fps_list):
    # ... (previous plot generation code) ...
    all_stream_fps = list(stream_fps_list) # Get data from Manager list
    all_fps = []
    all_cpu = []
    all_gpu = []
    time_stats = {
        'frame_read': 0.0, 'detection_tracker': 0.0, 'detection_processing': 0.0,
        'detection_drawing': 0.0, 'segmentation': 0.0, 'segmentation_drawing': 0.0
    }

    total_frames = 0
    # Ensure metrics_list is iterated correctly (it's a list of lists)
    for process_metrics in metrics_list:
        for frame in process_metrics:
            total_frames += 1
            # ... (rest of the loop as before) ...
            if frame['total'] > 0: all_fps.append(1 / frame['total'])
            else: all_fps.append(0)
            all_cpu.append(frame['cpu'])
            all_gpu.append(frame['gpu'])
            for key in time_stats: time_stats[key] += frame.get(key, 0)

    # *** FIX for ZeroDivisionError ***
    if total_frames == 0:
        print("No frames were processed. Cannot generate performance plots.")
        # Create dummy plot or just return
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'No frames processed.\nCannot generate plots.',
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.savefig('detailed_performance_analysis_FAILED.png')
        plt.close()
        return

    avg_fps = np.mean(all_fps) if all_fps else 0
    avg_cpu = np.mean(all_cpu) if all_cpu else 0
    avg_gpu = np.mean(all_gpu) if all_gpu else 0
    avg_stream_fps = np.mean([fps for fps in all_stream_fps if fps > 0]) if any(fps > 0 for fps in all_stream_fps) else 0 # Avoid averaging zeros if some streams failed? Or just np.mean(all_stream_fps)

    total_time = sum(time_stats.values())
    time_percentages = {k: (v / total_time * 100) if total_time > 0 else 0 for k, v in time_stats.items()}


    stream_fps_text = "\n".join([f"Stream {i}: {fps:.2f} FPS" for i, fps in enumerate(all_stream_fps)])

    # Ensure keys exist before formatting
    summary_text = f"""Performance Summary:
        -------------------
        Average End-to-End Stream FPS: {avg_stream_fps:.2f}
        Individual Stream FPS:
        {stream_fps_text}

        Average Per-Frame FPS: {avg_fps:.2f}
        Average CPU Usage: {avg_cpu:.2f}% (Note: CPU/GPU metrics not collected in current code)
        Average GPU Usage: {avg_gpu:.2f}% (Note: CPU/GPU metrics not collected in current code)

        Time Distribution (% of total processing time):
        - Frame Reading: {time_percentages.get('frame_read', 0):.1f}%
        - Detection (Model): {time_percentages.get('detection_tracker', 0):.1f}%
        - Detection (Processing): {time_percentages.get('detection_processing', 0):.1f}%
        - Detection (Drawing): {time_percentages.get('detection_drawing', 0):.1f}%
        - Segmentation (Model): {time_percentages.get('segmentation', 0):.1f}%
        - Segmentation (Drawing): {time_percentages.get('segmentation_drawing', 0):.1f}%

        Absolute Times per Frame (ms):
        - Frame Reading: {time_stats.get('frame_read', 0)/total_frames*1000:.2f}
        - Detection Model: {time_stats.get('detection_tracker', 0)/total_frames*1000:.2f}
        - Detection Processing: {time_stats.get('detection_processing', 0)/total_frames*1000:.2f}
        - Detection Drawing: {time_stats.get('detection_drawing', 0)/total_frames*1000:.2f}
        - Segmentation Model: {time_stats.get('segmentation', 0)/total_frames*1000:.2f}
        - Segmentation Drawing: {time_stats.get('segmentation_drawing', 0)/total_frames*1000:.2f}
        """

    # --- Plotting code (ensure labels match time_stats keys) ---
    plt.figure(figsize=(20, 15))

    # Summary text
    plt.subplot(3, 3, 1)
    plt.axis('off')
    plt.text(0, 0.1, summary_text, fontfamily='monospace', fontsize=9, va='bottom') # Adjust text position

    # FPS plot
    plt.subplot(3, 3, 2)
    if all_fps:
        plt.plot(all_fps, color='blue')
        plt.ylim(0, max(all_fps)*1.2 if all_fps else 30)
    plt.title('Frame Rate Over Time')
    plt.xlabel('Frame Number (Aggregated)')
    plt.ylabel('FPS')
    plt.grid(True)

    # CPU/GPU plot (Data currently not collected)
    plt.subplot(3, 3, 3)
    if all_cpu: plt.plot(all_cpu, label='CPU Usage', color='green')
    if all_gpu: plt.plot(all_gpu, label='GPU Usage', color='red')
    plt.title('System Resource Usage (Not Collected)')
    plt.xlabel('Frame Number (Aggregated)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)

    # Time distribution pie chart
    plt.subplot(3, 3, 4)
    valid_time_stats = {k: v for k,v in time_percentages.items() if v > 0} # Filter zero values
    labels = list(valid_time_stats.keys())
    sizes = list(valid_time_stats.values())
    if sizes:
        # Find the index of 'frame_read' for explode, handle if missing
        try: explode_idx = labels.index('frame_read')
        except ValueError: explode_idx = -1
        explode = [0.1 if i == explode_idx else 0 for i in range(len(labels))]

        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=140)
        plt.title('Processing Time Distribution (%)')
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, 'No timing data', ha='center', va='center')


    # Detailed timing plot (ms)
    plt.subplot(3, 3, 5)
    labels_detail = list(time_stats.keys()) # Use all labels for consistency
    times_ms = [time_stats.get(k, 0)/total_frames*1000 for k in labels_detail]
    x_pos = np.arange(len(labels_detail))
    plt.bar(x_pos, times_ms, align='center', alpha=0.7)
    plt.xticks(x_pos, labels_detail, rotation=45, ha='right')
    plt.ylabel('Time (ms)')
    plt.title('Average Time per Processing Stage')
    plt.grid(True)
    plt.ylim(bottom=0)

    # Cumulative time plot
    plt.subplot(3, 3, 6)
    cumulative_times = np.cumsum(times_ms)
    plt.plot(x_pos, cumulative_times, marker='o', color='purple') # Plot against x_pos
    plt.xticks(x_pos, labels_detail, rotation=45, ha='right')
    plt.ylabel('Cumulative Time (ms)')
    plt.title('Cumulative Processing Time')
    plt.grid(True)
    plt.ylim(bottom=0)


    # Stream FPS plot
    plt.subplot(3, 3, 7)
    if all_stream_fps:
      plt.bar(range(len(all_stream_fps)), all_stream_fps, color='cyan')
      plt.xticks(range(len(all_stream_fps)))
    plt.title('End-to-End FPS per Stream')
    plt.xlabel('Stream ID')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.ylim(bottom=0)


    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Add rect to prevent title overlap
    plt.suptitle("Detailed Performance Analysis", fontsize=16) # Add overall title
    plt.savefig('detailed_performance_analysis_multithread.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Performance plots saved as 'detailed_performance_analysis_multithread.png'")


# --- process_single_stream_thread function ---
def process_single_stream_thread(stream_source, stream_id, process_id,
                                 metrics_list, # The shared manager list for detailed frame metrics
                                 stream_fps_list, # The shared manager list for final FPS per stream
                                 detection_model, # Pass the loaded models
                                 segmentation_model): # Pass the loaded models
    """Handles processing for one stream within a thread."""
    thread_local_metrics = [] # Collect metrics locally before appending to shared list
    try:
        print(f'[Proc {process_id}/Thread {stream_id}] Starting stream {stream_id}')
        # NOTE: Models are passed in, assuming they are thread-safe or handled appropriately

        cap = ffmpegcv.VideoCaptureNV(stream_source, pix_fmt='bgr24', resize=(1280, 720))
        # Check if video opened successfully
        if not cap.isOpened():
             print(f"[Proc {process_id}/Thread {stream_id}] Error: Could not open video stream: {stream_source}")
             stream_fps_list[stream_id] = 0.0 # Record failure
             # Append an empty list or failure marker to metrics_list for this thread?
             # metrics_list.append([]) # Simplest option
             return # Exit thread

        frame_id = 0
        start_time = time.time()

        while True:
            frame_times = {
                'frame_read': 0.0, 'detection': 0.0, 'detection_tracker': 0.0,
                'detection_processing': 0.0, 'detection_drawing': 0.0,
                'segmentation': 0.0, 'segmentation_drawing': 0.0,
                'total': 0.0, 'cpu': 0.0, 'gpu': 0.0 # cpu/gpu still not measured here
            }

            s = time.time()
            ret, frame = cap.read()
            frame_times['frame_read'] = time.time() - s

            if not ret:
                # print(f"[Proc {process_id}/Thread {stream_id}] End of stream.") # Optional log
                break

            # --- Model processing (ensure models are used correctly) ---
            # IMPORTANT: Add locking here if model inference isn't thread-safe
            # e.g., with process_lock:
            #          results = detection_model.track(...)
            with torch.no_grad(): # Good practice
                frame_copy = frame.copy() # Work on a copy if models modify in-place

                # --- Uncomment and adapt if running models ---
                # det_start = time.time()
                # track_result = detection_model.track(frame=frame_copy, frame_id=frame_id)
                # potato_images, potato_boxes, tracker_time, proc_time, draw_time = track_result
                # frame_times['detection_tracker'] = tracker_time
                # frame_times['detection_processing'] = proc_time
                # frame_times['detection_drawing'] = draw_time
                # frame_times['detection'] = time.time() - det_start # Total detection time

                # seg_start = time.time()
                # if potato_images: # Check if there's anything to segment
                #     # Ensure tracked_sizes is handled correctly (maybe belongs to detector?)
                #     detection_model.tracked_sizes, frame_copy = segmentation_model.process_batch(
                #         potato_images,
                #         potato_boxes,
                #         detection_model.tracked_sizes, # Pass the correct state object
                #         frame_copy # Pass the potentially modified frame
                #     )
                # frame_times['segmentation'] = time.time() - seg_start
                # # Assuming segmentation_drawing time is part of process_batch or negligible
                # frame_times['segmentation_drawing'] = 0.0 # Or measure if separate

            # Calculate total time for this frame based on operations performed
            frame_times['total'] = sum(frame_times[k] for k in ['frame_read', 'detection', 'segmentation'])
            # Add other measured times if needed

            thread_local_metrics.append(frame_times)
            frame_id += 1
            # Optional: Add a small sleep if CPU is still too high just from looping
            # time.sleep(0.001)

        end_time = time.time()
        total_time = end_time - start_time

        # Handle case where video is valid but has 0 frames or processes instantly
        if frame_id > 0 and total_time > 0:
             stream_fps = frame_id / total_time
        elif frame_id > 0 and total_time == 0:
             stream_fps = float('inf') # Or some large number / special marker
        else: # frame_id == 0
             stream_fps = 0.0

        # --- Store results ---
        metrics_list.append(thread_local_metrics) # Append this thread's metrics (list of dicts)
        stream_fps_list[stream_id] = stream_fps # Update the specific stream's FPS using its original index

        print(f'[Proc {process_id}/Thread {stream_id}] Finished. Frames: {frame_id}, Time: {total_time:.2f}s, FPS: {stream_fps:.2f}')

    except Exception as e:
        import traceback
        print(f"[Proc {process_id}/Thread {stream_id}] Error: {str(e)}")
        print(traceback.format_exc()) # Print full traceback for thread errors
        stream_fps_list[stream_id] = 0.0 # Mark as failed
        metrics_list.append([]) # Append empty metrics on error

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        # Release other resources if needed
        print(f"[Proc {process_id}/Thread {stream_id}] Released resources")

# --- worker_process function ---
def worker_process(process_id, streams_for_this_process, assigned_stream_ids,
                   metrics_list, stream_fps_list): # Pass shared lists
    """A single process that manages multiple stream threads."""
    try:
        print(f"[Proc {process_id}] Initializing...")
        # Initialize CUDA context PER PROCESS
        # import pycuda.autoinit # May not be needed if torch handles it
        # torch.cuda.init()      # May not be needed if torch handles it
        # It's often sufficient that torch uses CUDA correctly later

        # --- Load models ONCE per process ---
        # This saves significant GPU memory
        print(f'[Proc {process_id}] Loading detection model...')
        # Ensure paths are correct
        detection_model = PotatoDetector(
            model_path='Models/model_det_test/best.engine',
            camera_id=f"proc_{process_id}",
            task='detect',
            tracker_config='bytetrack_custom.yaml',
            segmentation_area=[432, 0, 779, 720],
            track_ids=set(),
            warmup_image='frame.jpg' # Ensure frame.jpg exists
        )
        print(f'[Proc {process_id}] Loading segmentation model...')
        segmentation_model = PotatoSegmentation(
            model_path='Models/model_seg_fp16/best_yoloseg.engine',
            ratio=0.02997,
            warmup_image='box.jpg' # Ensure box.jpg exists
        )
        print(f"[Proc {process_id}] Models loaded.")

        threads = []
        # Create a lock if models aren't thread-safe for inference
        # process_lock = threading.Lock()

        for i, stream_source in enumerate(streams_for_this_process):
            stream_id = assigned_stream_ids[i] # Get the original global stream ID
            thread = threading.Thread(
                target=process_single_stream_thread,
                args=(stream_source, stream_id, process_id,
                      metrics_list, # Pass manager list
                      stream_fps_list, # Pass manager list
                      detection_model, # Pass loaded model
                      segmentation_model), # Pass loaded model
                      # Add lock if needed: process_lock=process_lock)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads in this process to complete
        for thread in threads:
            thread.join()

        # Metrics are directly appended by threads to the manager list

        print(f"[Proc {process_id}] All threads finished.")

    except Exception as e:
        import traceback
        print(f"[Proc {process_id}] Error: {str(e)}")
        print(traceback.format_exc()) # Print full traceback for process errors
        # Mark all streams assigned to this process as failed
        for i in range(len(assigned_stream_ids)):
             stream_id = assigned_stream_ids[i]
             if stream_fps_list[stream_id] == 0.0: # Avoid overwriting if a thread finished ok before process crash
                 stream_fps_list[stream_id] = -1.0 # Use -1 or similar to indicate process-level failure
             metrics_list.append([]) # Add empty metrics

    finally:
        # Clean up process-level resources (models etc.)
        # Deleting might not be strictly necessary if process exits cleanly, but good practice
        if 'detection_model' in locals(): del detection_model
        if 'segmentation_model' in locals(): del segmentation_model
        print(f"[Proc {process_id}] Cleaned up process resources.")


# --- main function ---
def main(num_processes=8):
    global ALL_STREAMS, metrics_list, stream_fps_list # Access the lists defined in __main__ block

    num_total_streams = len(ALL_STREAMS)
    if num_processes <= 0:
        num_processes = os.cpu_count() or 1
    num_processes = min(num_processes, num_total_streams)

    streams_per_process_list = [[] for _ in range(num_processes)]
    assigned_stream_ids_list = [[] for _ in range(num_processes)]
    for i, stream in enumerate(ALL_STREAMS):
        process_idx = i % num_processes
        streams_per_process_list[process_idx].append(stream)
        assigned_stream_ids_list[process_idx].append(i)

    processes = []
    print(f"Starting {num_processes} processes to handle {num_total_streams} streams...")

    for i in range(num_processes):
        p = mp.Process(
            target=worker_process,
            args=(i, streams_per_process_list[i], assigned_stream_ids_list[i],
                  metrics_list, stream_fps_list) # Pass manager lists to the process
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished. Generating plots...")
    # Pass copies from the manager list to plotting function
    generate_plots(list(metrics_list), list(stream_fps_list))

# --- Main execution Guard ---
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # *** FIX for RuntimeError: Move Manager creation here ***
    manager = mp.Manager()
    metrics_list = manager.list() # For detailed frame metrics [[{frame1},{frame2}], [{frame1}]]
    stream_fps_list = manager.list([0.0] * len(ALL_STREAMS)) # Pre-initialize for indexed assignment

    # Set start method explicitly for CUDA+multiprocessing
    # Should be called only once, and ideally right at the start of the if __name__ block
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        print(f"Note: multiprocessing start method already set or failed: {e}")


    # Initialize TensorRT plugins once in the main process
    # May need to be done in worker_process if plugins rely on process-specific CUDA contexts
    try:
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        print("TensorRT plugins initialized.")
    except Exception as e:
        print(f"Warning: TensorRT plugin initialization failed: {e}")


    # --- Experiment with the number of processes ---
    main(num_processes=8) # Try 8, 10, 12, 16 etc.