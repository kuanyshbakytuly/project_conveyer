import ffmpegcv
import threading
import queue
import torch
import time
import pynvml
import psutil
from datetime import datetime
import atexit
import numpy as np
import cv2
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")

# Configuration
NUM_VIDEOS = 20  # Start with 1 video for debugging
BATCH_SIZE = 1  # Reduced for testing
VIDEO_PATHS = ['output.mp4'] * NUM_VIDEOS
MONITOR_INTERVAL = 2  # More frequent monitoring

class SystemMonitor:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.running = False
        self.thread = None
        self.handle = None
        self.initialize_nvml()
        atexit.register(self.cleanup)
        
    def initialize_nvml(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as e:
            print(f"NVML initialization failed: {e}")
            self.handle = None
            
    def start(self):
        if self.handle is None:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def _monitor_loop(self):
        while self.running and self.handle is not None:
            try:
                self.report_stats()
                time.sleep(MONITOR_INTERVAL)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
            
    def get_gpu_stats(self):
        if self.handle is None:
            return {'gpu_util': 0, 'mem_util': 0, 'mem_used': 0, 'mem_total': 0, 'temp': 0}
            
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return {
                'gpu_util': util.gpu,
                'mem_util': 100 * mem.used / mem.total,
                'mem_used': mem.used / (1024 ** 2),
                'mem_total': mem.total / (1024 ** 2),
                'temp': temp
            }
        except Exception as e:
            print(f"GPU stats error: {e}")
            return {'gpu_util': 0, 'mem_util': 0, 'mem_used': 0, 'mem_total': 0, 'temp': 0}
    
    def get_cpu_stats(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            return {
                'cpu_util': cpu_percent,
                'cpu_mem_used': mem.used / (1024 ** 2),
                'cpu_mem_total': mem.total / (1024 ** 2)
            }
        except Exception as e:
            print(f"CPU stats error: {e}")
            return {'cpu_util': 0, 'cpu_mem_used': 0, 'cpu_mem_total': 0}
    
    def report_stats(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        gpu_stats = self.get_gpu_stats()
        cpu_stats = self.get_cpu_stats()
        
        stats_msg = (
            f"[{timestamp}] System Stats:\n"
            f"GPU {self.gpu_id}: {gpu_stats['gpu_util']}% util | "
            f"{gpu_stats['mem_util']:.1f}% mem ({gpu_stats['mem_used']:.1f}/{gpu_stats['mem_total']:.1f} MB) | "
            f"{gpu_stats['temp']}°C\n"
            f"CPU: {cpu_stats['cpu_util']}% util | "
            f"Memory: {cpu_stats['cpu_mem_used']:.1f}/{cpu_stats['cpu_mem_total']:.1f} MB"
        )
        print(stats_msg)
    
    def cleanup(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        try:
            if pynvml.nvmlShutdown:
                pynvml.nvmlShutdown()
        except:
            pass

class VideoStreamer:
    def __init__(self, video_path, gpu_id=0):
        self.video_path = video_path
        self.gpu_id = gpu_id
        self.frame_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        self.thread = None
        self.vid = None
        
    def start_streaming(self):
        self.thread = threading.Thread(target=self._stream_frames, daemon=True)
        self.thread.start()
        
    def _stream_frames(self):
        try:
            self.vid = ffmpegcv.VideoCapture(self.video_path)
            
            while not self.stop_event.is_set():
                ret, frame = self.vid.read()
                if not ret:
                    break
                
                if frame is not None and frame.size > 0:
                    try:
                        frame = np.ascontiguousarray(frame)
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                            
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(f'cuda:{self.gpu_id}')
                        self.frame_queue.put(frame_tensor)
                    except Exception as e:
                        print(f"Frame conversion error: {e}")
                        break
                else:
                    print("Warning: Empty frame received")
                    break
                    
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            if hasattr(self, 'vid') and self.vid is not None:
                self.vid.release()
            self.frame_queue.put(None)
            
    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if hasattr(self, 'vid') and self.vid is not None:
            self.vid.release()

class BatchPredictor:
    def __init__(self, model, gpu_id=0):
        self.model = model
        self.gpu_id = gpu_id
        self.streamers = []
        self.frame_count = 0
        self.start_time = None
        self.monitor = SystemMonitor(gpu_id)
        self.last_report_time = time.time()
        
    def initialize(self):
        self.start_time = time.time()
        self.monitor.start()
        
        for path in VIDEO_PATHS[:NUM_VIDEOS]:
            streamer = VideoStreamer(path, self.gpu_id)
            streamer.start_streaming()
            self.streamers.append(streamer)
            
    def process_batches(self):
        batch_frames = []
        active_streams = len(self.streamers)
        
        while active_streams > 0:
            for i, streamer in enumerate(self.streamers):
                if streamer is None:
                    continue
                    
                frame = streamer.get_frame()
                if frame is None:
                    if not streamer.thread.is_alive():
                        self.streamers[i] = None
                        active_streams -= 1
                    continue
                
                batch_frames.append(frame)
                self.frame_count += 1
                
                if len(batch_frames) >= BATCH_SIZE:
                    try:
                        batch_tensor = torch.stack(batch_frames)
                        
                        with torch.no_grad():
                            start_process = time.time()
                            # predictions = self.model(batch_tensor)
                            time.sleep(0.01)
                            process_time = time.time() - start_process
                            
                        current_time = time.time()
                        if current_time - self.last_report_time >= 1.0:
                            elapsed = current_time - self.start_time
                            fps = self.frame_count / elapsed
                            print(f"\nFPS: {fps:.1f} | Process Time: {process_time*1000:.1f}ms | Frames: {self.frame_count}")
                            self.last_report_time = current_time
                            
                        yield len(batch_frames)
                    except Exception as e:
                        print(f"Batch processing error: {e}")
                    finally:
                        batch_frames.clear()
                    
        if batch_frames:
            try:
                batch_tensor = torch.stack(batch_frames)
                with torch.no_grad():
                    # predictions = self.model(batch_tensor)
                    time.sleep(0.01)
                yield len(batch_frames)
            except Exception as e:
                print(f"Final batch error: {e}")
            
    def shutdown(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n=== Final Stats ===")
        print(f"FPS: {fps:.1f}")
        print(f"Total frames: {self.frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        
        self.monitor.cleanup()
        for streamer in self.streamers:
            if streamer:
                streamer.stop()
        torch.cuda.empty_cache()

# def load_your_model(path):
#     model = YOLO(path, task='detect').to('cuda')
#     return model

# def warm_up_model(model, iterations=10):
#     dummy_input = torch.randn(1, 3, 640, 640, device='cuda')
#     for _ in range(iterations):
#         _ = model(dummy_input)
#     torch.cuda.synchronize()
#     print(f"Model warmed up with {iterations} iterations")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    
    try:
        print("Initializing model...")
        # model = load_your_model(path='Models/best_yolov8n1.pt')
        # warm_up_model(model)
        
        print("Starting predictor...")
        predictor = BatchPredictor(None, gpu_id=0)
        predictor.initialize()
        
        try:
            print("Processing started - Ctrl+C to stop")
            for batch_num, frame_count in enumerate(predictor.process_batches()):
                print(f"Batch {batch_num}: {frame_count} frames", end='\r')
                
        except KeyboardInterrupt:
            print("\nStopping early...")
        finally:
            predictor.shutdown()
            
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        torch.cuda.empty_cache()