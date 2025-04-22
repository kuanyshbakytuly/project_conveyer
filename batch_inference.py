# batch_inference.py
import ffmpegcv
import multiprocessing as mp
import torch
import time
from queue import Empty
from collections import defaultdict
import numpy as np
import cv2
from cv_api.src.detection.batch_detection import PotatoDetector
from cv_api.src.segmentation.batch_segmentation import PotatoSegmentation

# Configuration
BATCH_SIZE = 8
MAX_QUEUE_SIZE = 500
NUM_POSTPROCESS_WORKERS = 8
GPU_MEMORY_FRACTION = 0.85
STREAM_TIMEOUT = 5.0  # Seconds
SEGMENTATION_AREAS = [
    [1295, 1, 2338, 2154]  # Define for each stream
] * 5  # For 10 streams

class FrameReader(mp.Process):
    def __init__(self, rtsp_url, output_queue):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.output_queue = output_queue
        self.daemon = True
        self.stop_flag = mp.Event()

    def run(self):
        cap = ffmpegcv.VideoCaptureNV(self.rtsp_url, pix_fmt='bgr24')
        while not self.stop_flag.is_set():
            if True:
                start_time = time.time()
                ret, frame = cap.read()
                if ret:
                    self.output_queue.put((
                        time.time() - start_time, 
                        frame, 
                        time.time()
                    ), block=True, timeout=STREAM_TIMEOUT)
                else:
                    break
            # except Exception as e:
            #     print(f"FrameReader error: {str(e)}")
            #     break
        cap.release()

    def stop(self):
        self.stop_flag.set()

class BatchProcessor(mp.Process):
    def __init__(self, input_queues, output_queue, segmentation_areas):
        super().__init__()
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.segmentation_areas = segmentation_areas
        self.daemon = True
        self.detectors = []
        self.segmentation = None

    def initialize_models(self):
        # Initialize detector for each stream with its segmentation area
        # for sa in self.segmentation_areas:
        #     self.detectors.append(PotatoDetector(
        #         model_path='Models/model_det_test/best.pt',
        #         warmup_image='frame.jpg',
        #         tracker_config='bytetrack_custom.yaml',
        #         segmentation_area=sa
        #     ))
        
        # self.segmentation = PotatoSegmentation(
        #     model_path='Models/model_seg_fp32/best_yoloseg.engine',
        #     warmup_image='box.jpg',
        #     ratio=0.00999000999000999
        # )
        return

    def run(self):
        self.initialize_models()
        batch_buffer = []
        last_flush = time.time()
        
        while True:
            # Collect frames from all streams
            now = time.time()
            for q_idx, q in enumerate(self.input_queues):
                try:
                    while True:
                        read_time, frame, timestamp = q.get_nowait()
                        batch_buffer.append((q_idx, read_time, frame, timestamp))
                except Empty:
                    continue

            # Process batch if: full, timeout, or stale frames
            if (len(batch_buffer) >= BATCH_SIZE or 
                (time.time() - last_flush > 0.03)):
                #or (now - batch_buffer[0][3] > 0.1)):
                time.sleep(0.02)
                #self.process_batch(batch_buffer)
                batch_buffer = []
                last_flush = time.time()

    def process_batch(self, batch):
        stream_batches = defaultdict(list)
        for q_idx, read_time, frame, timestamp in batch:
            stream_batches[q_idx].append((read_time, frame, timestamp))

        for stream_idx, stream_batch in stream_batches.items():
            frames = [f[1] for f in stream_batch]
            detector = self.detectors[stream_idx]
            
            # Get filtered detections within segmentation area
            print(len(frames), frames[0].shape)
            det_results = detector.batch_track(frames)
            
            # Process segmentation only for valid boxes
            seg_results = self.segmentation.process_batch(frames, det_results)
            
            for i, (read_time, frame, timestamp) in enumerate(stream_batch):
                self.output_queue.put({
                    'stream_idx': stream_idx,
                    'frame': frame,
                    'detections': det_results[i],
                    'segmentation': seg_results[i],
                    'timing': {
                        'frame_read': read_time,
                        'detection': det_results[i]['processing_time'],
                        'segmentation': seg_results[i]['processing_time']
                    }
                })

class PostProcessor(mp.Process):
    def __init__(self, input_queue, output_queues, metric_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.metric_queue = metric_queue
        self.daemon = True

    def run(self):
        while True:
            try:
                result = self.input_queue.get(timeout=1)
                self.process_result(result)
            except Empty:
                continue

    def process_result(self, result):
        stream_idx = result['stream_idx']
        frame = result['frame']
        
        # Draw operations
        draw_start = time.time()
        
        # Draw detection boxes
        for det in result['detections']['boxes']:
            x1, y1, x2, y2 = det['coords']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {det['id']}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw segmentation masks
        for seg in result['segmentation']['masks']:
            cv2.fillPoly(frame, [seg['contour']], (0, 255, 0, 0.3))
        
        # Draw segmentation area boundary
        sa = SEGMENTATION_AREAS[stream_idx]
        cv2.rectangle(frame, (sa[0], sa[1]), (sa[2], sa[3]), (0, 0, 255), 2)
        
        draw_time = time.time() - draw_start
        
        # Send to output
        self.output_queues[stream_idx].put(frame)
        
        # Collect metrics
        self.metric_queue.put({
            **result['timing'],
            'detection_drawing': draw_time,
            'total': sum(result['timing'].values()) + draw_time
        })

def generate_performance_report(metric_queue):
    metrics = []
    while not metric_queue.empty():
        metrics.append(metric_queue.get())
    
    if not metrics:
        return
    
    # Calculate average metrics
    avg_fps = 1 / np.mean([m['total'] for m in metrics])
    avg_cpu = np.mean([m['cpu_usage'] for m in metrics])
    avg_gpu = np.mean([m['gpu_usage'] for m in metrics])
    
    print(f"\nPerformance Report:")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"CPU Usage: {avg_cpu:.1f}%")
    print(f"GPU Usage: {avg_gpu:.1f}%")
    print(f"Frames Processed: {len(metrics)}")
    print("-----------------------------\n")

def main():
    streams = ['video.mp4'] * 5  # 10 streams
    manager = mp.Manager()
    
    # Create queues
    input_queues = [manager.Queue(MAX_QUEUE_SIZE) for _ in streams]
    batch_queue = manager.Queue()
    metric_queue = manager.Queue()
    output_queues = [manager.Queue() for _ in streams]

    # Start components
    readers = [FrameReader(url, q) for url, q in zip(streams, input_queues)]
    processor = BatchProcessor(input_queues, batch_queue, SEGMENTATION_AREAS)
    post_processors = [PostProcessor(batch_queue, output_queues, metric_queue) 
                      for _ in range(NUM_POSTPROCESS_WORKERS)]

    for r in readers: r.start()
    processor.start()
    for p in post_processors: p.start()

    # Monitoring
    try:
        while True:
            time.sleep(5)
            generate_performance_report(metric_queue)
    except KeyboardInterrupt:
        for r in readers: r.stop()
        processor.terminate()
        for p in post_processors: p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    main()