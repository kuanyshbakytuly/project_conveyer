#!/usr/bin/env python3
"""
Example script showing how to use the RTSP-enabled video processor
This can be used for testing without Docker or for custom integrations
"""

import multiprocessing as mp
import cv2
import numpy as np
from typing import List
import time
import os

# Set up environment for testing
os.environ['NUM_STREAMS'] = '2'
os.environ['TARGET_FPS'] = '25'
os.environ['FRAME_SKIP'] = '2'

# Example RTSP URLs - replace with your cameras
RTSP_URLS = [
    "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101",
    "rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101",
]

# Set environment variables for processor
for i, url in enumerate(RTSP_URLS):
    os.environ[f'RTSP_URL_{i}'] = url


def display_frames(queue_id: int, frame_queue: mp.Queue):
    """Simple display function for processed frames"""
    window_name = f"Camera {queue_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    frame_count = 0
    fps_start = time.time()
    
    while True:
        try:
            # Get frame from queue
            frame_data = frame_queue.get(timeout=5.0)
            
            # Decode JPEG
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if frame is not None:
                # Add FPS counter
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - fps_start)
                    print(f"Camera {queue_id}: {fps:.2f} FPS")
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except mp.queues.Empty:
            print(f"Camera {queue_id}: No frames received for 5 seconds")
        except Exception as e:
            print(f"Camera {queue_id} display error: {e}")
            break
    
    cv2.destroyWindow(window_name)


def simple_processor(stream_url: str, stream_id: int, output_queue: mp.Queue):
    """Simplified processor for testing - no ML models"""
    import ffmpegcv
    
    print(f"Starting simple processor for stream {stream_id}: {stream_url}")
    
    cap = None
    reconnect_attempts = 0
    
    while reconnect_attempts < 5:
        try:
            # Open RTSP stream
            if stream_url.startswith('rtsp://'):
                # Use low latency mode
                cap = ffmpegcv.VideoCaptureStreamRT(stream_url)
            else:
                cap = cv2.VideoCapture(stream_url)
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream {stream_id}: Failed to read frame")
                    break
                
                # Simple processing - just add text
                text = f"Stream {stream_id} - Frame {frame_count}"
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Send to queue
                try:
                    output_queue.put_nowait(buffer.tobytes())
                except:
                    pass  # Queue full, skip
                
                frame_count += 1
                
                # Simulate frame skip
                if frame_count % 2 == 0:
                    time.sleep(0.01)  # Small delay
                    
        except Exception as e:
            print(f"Stream {stream_id} error: {e}")
        finally:
            if cap:
                cap.release()
                cap = None
        
        reconnect_attempts += 1
        print(f"Stream {stream_id}: Reconnecting (attempt {reconnect_attempts})...")
        time.sleep(2)
    
    print(f"Stream {stream_id}: Exiting after {reconnect_attempts} attempts")


def main():
    """Main function demonstrating RTSP processing"""
    print("RTSP Video Processing Example")
    print("=============================")
    print(f"Processing {len(RTSP_URLS)} streams")
    print("Press 'q' in any window to quit")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues and processes
    queues = []
    processes = []
    display_processes = []
    
    # Option 1: Use the full processor with ML models
    use_full_processor = False
    
    if use_full_processor:
        # Import the actual processor
        from processor import process_stream
        processor_func = process_stream
    else:
        # Use simple processor for testing
        processor_func = simple_processor
    
    # Start processing streams
    for i, url in enumerate(RTSP_URLS):
        # Create queue
        queue = mp.Queue(maxsize=5)
        queues.append(queue)
        
        # Start processor
        if use_full_processor:
            # Full processor expects GPU ID as last argument
            gpu_id = i % 2  # Distribute across 2 GPUs
            p = mp.Process(
                target=processor_func,
                args=(url, i, queue, gpu_id),
                daemon=True
            )
        else:
            # Simple processor
            p = mp.Process(
                target=processor_func,
                args=(url, i, queue),
                daemon=True
            )
        p.start()
        processes.append(p)
        
        # Start display process
        d = mp.Process(
            target=display_frames,
            args=(i, queue),
            daemon=True
        )
        d.start()
        display_processes.append(d)
    
    print("\nProcessing started. Press Ctrl+C to stop.")
    
    try:
        # Monitor processes
        while True:
            time.sleep(1)
            
            # Check if any process died
            for i, p in enumerate(processes):
                if not p.is_alive():
                    print(f"Process {i} died, restarting...")
                    processes[i] = mp.Process(
                        target=processor_func,
                        args=(RTSP_URLS[i], i, queues[i]),
                        daemon=True
                    )
                    processes[i].start()
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        for p in processes + display_processes:
            p.terminate()
        for p in processes + display_processes:
            p.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()