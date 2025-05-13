import ffmpegcv
import time
import threading
from multiprocessing import Pool

def read_frames(stream):

    path, gpu_id = stream
    cap = ffmpegcv.VideoCaptureNV(path, pix_fmt='bgr24', resize=(1280, 720), gpu=gpu_id)
    
    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    end_time = time.time()
    cap.release()
    return count, start_time, end_time

def run_multithreaded(streams):
    results = []
    threads = []
    
    def worker(stream):
        result = read_frames(stream)
        results.append(result)
    
    for stream in streams:
        t = threading.Thread(target=worker, args=(stream,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    counts, starts, ends = zip(*results)
    total_frames = sum(counts)
    total_start = min(starts)
    total_end = max(ends)
    duration = total_end - total_start
    fps = total_frames / duration
    print(f"Multithreaded FPS: {fps:.2f}")

def run_multiprocessed(streams):
    with Pool(len(streams)) as pool:
        results = pool.map(read_frames, streams)
    
    counts, starts, ends = zip(*results)
    total_frames = sum(counts)
    total_start = min(starts)
    total_end = max(ends)
    duration = total_end - total_start
    fps = total_frames / duration
    print(f"Multiprocessed FPS: {fps:.2f}")

if __name__ == '__main__':
    streams = [('video.mov', i%2) for i in range(20)]
    
    print("Testing multithreading...")
    run_multithreaded(streams)
    
    print("Testing multiprocessing...")
    run_multiprocessed(streams)