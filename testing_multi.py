import ffmpegcv
import time
import threading
from multiprocessing import Pool

def read_frames(stream):
    # Initialize capture first (this is part of the measured time)
    cap = ffmpegcv.toCUDA(
        ffmpegcv.VideoCaptureNV(stream, pix_fmt='nv12', resize=(640, 640)), 
        tensor_format='chw'
    )
    
    # Start timing AFTER initialization
    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    end_time = time.time()
    cap.release()
    return count, start_time, end_time  # Return timing per worker

def run_multithreaded(streams):
    results = []
    threads = []
    
    def worker(stream):
        result = read_frames(stream)
        results.append(result)
    
    # Start all threads
    for stream in streams:
        t = threading.Thread(target=worker, args=(stream,))
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Calculate true parallel duration
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
    
    # Calculate true parallel duration
    counts, starts, ends = zip(*results)
    total_frames = sum(counts)
    total_start = min(starts)
    total_end = max(ends)
    duration = total_end - total_start
    fps = total_frames / duration
    print(f"Multiprocessed FPS: {fps:.2f}")

if __name__ == '__main__':
    streams = ['video.mp4'] * 20  # Replace with actual paths
    
    print("Testing multithreading...")
    run_multithreaded(streams)
    
    print("Testing multiprocessing...")
    run_multiprocessed(streams)