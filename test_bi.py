# batch_inference.py (FPS Test Version)
import ffmpegcv
import multiprocessing as mp
import time
from queue import Empty
import numpy as np

# Configuration
NUM_STREAMS = 10
MAX_QUEUE_SIZE = 500
STREAM_TIMEOUT = 5.0  # Seconds

class FrameReader(mp.Process):
    def __init__(self, rtsp_url, metric_queue):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.metric_queue = metric_queue
        self.daemon = True
        self.stop_flag = mp.Event()

    def run(self):
        cap = ffmpegcv.VideoCaptureNV(self.rtsp_url, pix_fmt='bgr24')
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_flag.is_set():
            try:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    # Calculate instantaneous FPS every 100 frames
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        self.metric_queue.put({
                            'fps': frame_count / elapsed,
                            'frame_count': frame_count,
                            'elapsed': elapsed
                        })
                else:
                    break
            except Exception as e:
                print(f"FrameReader error: {str(e)}")
                break
        
        # Final FPS calculation
        total_elapsed = time.time() - start_time
        self.metric_queue.put({
            'final_fps': frame_count / total_elapsed,
            'total_frames': frame_count,
            'total_elapsed': total_elapsed
        })
        cap.release()

    def stop(self):
        self.stop_flag.set()

def generate_performance_report(metric_queue):
    metrics = []
    while not metric_queue.empty():
        metrics.append(metric_queue.get())
    
    if not metrics:
        return
    
    # Calculate averages
    total_frames = 0
    total_elapsed = 0
    final_results = []
    
    for m in metrics:
        if 'final_fps' in m:
            final_results.append(m)
        else:
            total_frames += m.get('frame_count', 0)
            total_elapsed += m.get('elapsed', 0)
    
    print("\nLive Performance:")
    if total_elapsed > 0:
        print(f"Current FPS: {total_frames / total_elapsed:.2f}")
    
    for res in final_results:
        print(f"\nFinal Results for Stream:")
        print(f"Total Frames: {res['total_frames']}")
        print(f"Total Time: {res['total_elapsed']:.2f} seconds")
        print(f"Average FPS: {res['final_fps']:.2f}")

def main():
    streams = ['video.mp4'] * NUM_STREAMS
    manager = mp.Manager()
    metric_queue = manager.Queue()

    # Start frame readers
    readers = [FrameReader(url, metric_queue) for url in streams]
    for r in readers:
        r.start()

    # Monitoring
    try:
        while True:
            time.sleep(5)
            generate_performance_report(metric_queue)
    except KeyboardInterrupt:
        for r in readers:
            r.stop()
        for r in readers:
            r.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()