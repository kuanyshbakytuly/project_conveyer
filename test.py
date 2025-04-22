import ffmpegcv
import time
import numpy as np

def safe_stress_test(num_streams, width, height, test_duration=10):
    print(f"\n=== Testing {num_streams} streams at {width}x{height} ===")
    
    caps = []
    for i in range(num_streams):
        try:
            # Using only documented parameters
            cap = ffmpegcv.VideoCaptureNV(
                'video.mp4',
                pix_fmt='bgr24',
                resize=(width, height),
                gpu=0  # Explicitly select GPU 0
            )
            
            # Verify stream with timeout
            start = time.time()
            while time.time() - start < 2.0:  # 2-second timeout
                ret, _ = cap.read()
                if ret:
                    break
            else:
                print(f"Stream {i} failed initial read")
                cap.release()
                continue
                
            caps.append(cap)
            time.sleep(0.1)  # Stagger initialization
        except Exception as e:
            print(f"Stream {i} initialization failed: {str(e)}")
    
    if not caps:
        print("No working streams initialized")
        return

    # Warm-up period
    time.sleep(1)
    
    # Test with proper error handling
    start = time.time()
    frame_counts = np.zeros(len(caps), dtype=int)
    
    try:
        while time.time() - start < test_duration:
            for i, cap in enumerate(caps):
                if cap is None:
                    continue
                    
                try:
                    ret, _ = cap.read()
                    if ret:
                        frame_counts[i] += 1
                    else:
                        print(f"Stream {i} ended")
                        caps[i] = None
                except Exception as e:
                    print(f"Stream {i} error: {str(e)}")
                    caps[i] = None
            
            # Early exit if all streams failed
            if all(cap is None for cap in caps):
                break
    finally:
        for cap in caps:
            if cap is not None:
                cap.release()
    
    # Calculate statistics
    working_streams = sum(1 for cap in caps if cap is not None)
    total_frames = frame_counts.sum()
    
    if total_frames > 0:
        total_fps = total_frames / test_duration
        avg_fps = total_fps / num_streams
        working_fps = total_fps / working_streams if working_streams > 0 else 0
        
        print(f"Working streams: {working_streams}/{num_streams}")
        print(f"Total FPS: {total_fps:.1f}")
        print(f"Average FPS per stream: {avg_fps:.1f}")
        if working_streams != num_streams:
            print(f"FPS per working stream: {working_fps:.1f}")
    else:
        print("Test failed - no frames decoded")

# Progressive test with recommended resolutions
if __name__ == '__main__':
    print("=== NVDEC Capacity Stress Test ===")
    
    # Test sequence (resolution tailored for A4000)
    tests = [
        (4, 2300, 2154),  # Full resolution
        (6, 1920, 1080),   # 1080p
        (8, 1280, 720),    # 720p
        (12, 854, 480),    # 480p
        (16, 640, 360)     # 360p
    ]
    
    for num_streams, width, height in tests:
        safe_stress_test(num_streams, width, height)
        time.sleep(5)  # Cool-down period