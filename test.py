#!/usr/bin/env python3
"""
RTSP Camera Connection Test Script
Tests all configured RTSP cameras before running the main pipeline
"""

import os
import sys
import time
import cv2
import ffmpegcv
from datetime import datetime
from typing import List, Tuple, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import configuration
try:
    from config import processor_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found, using environment variables only")


def test_opencv_connection(url: str, timeout: int = 10) -> Tuple[bool, str, Dict]:
    """Test RTSP connection using OpenCV"""
    result = {"method": "OpenCV", "success": False, "error": "", "details": {}}
    
    try:
        print(f"Testing with OpenCV: {hide_password(url)}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            result["error"] = "Failed to open stream"
            return False, "Failed to open stream", result
        
        # Try to read a frame
        start_time = time.time()
        ret, frame = cap.read()
        read_time = time.time() - start_time
        
        if ret and frame is not None:
            result["success"] = True
            result["details"] = {
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "read_time": f"{read_time:.3f}s",
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "codec": cap.get(cv2.CAP_PROP_FOURCC)
            }
            cap.release()
            return True, "Success", result
        else:
            result["error"] = "Failed to read frame"
            cap.release()
            return False, "Failed to read frame", result
            
    except Exception as e:
        result["error"] = str(e)
        return False, str(e), result


def test_ffmpegcv_connection(url: str, timeout: int = 10) -> Tuple[bool, str, Dict]:
    """Test RTSP connection using ffmpegcv"""
    result = {"method": "ffmpegcv", "success": False, "error": "", "details": {}}
    
    try:
        print(f"Testing with ffmpegcv: {hide_password(url)}")
        
        # Determine stream type
        if url.startswith('rtsp://'):
            # Use low-latency RTSP capture
            cap = ffmpegcv.VideoCaptureStreamRT(url)
            result["details"]["mode"] = "RTSP Low Latency"
        else:
            # HTTP/HLS stream
            cap = ffmpegcv.VideoCaptureStream(url)
            result["details"]["mode"] = "HTTP Stream"
        
        # Try to read frames
        start_time = time.time()
        ret, frame = cap.read()
        read_time = time.time() - start_time
        
        if ret and frame is not None:
            result["success"] = True
            result["details"].update({
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "read_time": f"{read_time:.3f}s"
            })
            
            # Test no-buffer mode
            try:
                cap_nobuffer = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, url)
                ret2, frame2 = cap_nobuffer.read()
                if ret2:
                    result["details"]["no_buffer_mode"] = "Supported"
                cap_nobuffer.release()
            except:
                result["details"]["no_buffer_mode"] = "Not supported"
            
            cap.release()
            return True, "Success", result
        else:
            result["error"] = "Failed to read frame"
            cap.release()
            return False, "Failed to read frame", result
            
    except Exception as e:
        result["error"] = str(e)
        return False, str(e), result


def hide_password(url: str) -> str:
    """Hide password in RTSP URL for display"""
    if '@' in url:
        parts = url.split('@')
        proto_user = parts[0].split('//')
        if len(proto_user) > 1:
            user_parts = proto_user[1].split(':')
            if len(user_parts) > 1:
                return f"{proto_user[0]}://{user_parts[0]}:****@{parts[1]}"
    return url


def test_camera(camera_id: int, url: str, methods: List[str]) -> Dict:
    """Test a single camera with specified methods"""
    print(f"\n{'='*60}")
    print(f"Testing Camera {camera_id}: {hide_password(url)}")
    print(f"{'='*60}")
    
    results = {
        "camera_id": camera_id,
        "url": hide_password(url),
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test with each method
    for method in methods:
        if method == "opencv":
            success, message, details = test_opencv_connection(url)
            results["tests"]["opencv"] = details
        elif method == "ffmpegcv":
            success, message, details = test_ffmpegcv_connection(url)
            results["tests"]["ffmpegcv"] = details
    
    # Summary
    any_success = any(test.get("success", False) for test in results["tests"].values())
    results["overall_success"] = any_success
    
    if any_success:
        print(f"✓ Camera {camera_id}: Connection successful")
        for method, details in results["tests"].items():
            if details.get("success"):
                print(f"  - {method}: {details.get('details', {}).get('resolution', 'Unknown resolution')}")
    else:
        print(f"✗ Camera {camera_id}: Connection failed")
        for method, details in results["tests"].items():
            print(f"  - {method}: {details.get('error', 'Unknown error')}")
    
    return results


def get_rtsp_urls() -> List[str]:
    """Get RTSP URLs from configuration or environment"""
    urls = []
    
    # Try configuration file first
    if CONFIG_AVAILABLE:
        urls = processor_config.rtsp_urls
    
    # Check environment variables
    env_urls = []
    for i in range(100):
        url = os.environ.get(f'RTSP_URL_{i}')
        if url:
            env_urls.append(url)
        else:
            break
    
    # Check comma-separated list
    urls_list = os.environ.get('RTSP_URLS')
    if urls_list:
        env_urls = [url.strip() for url in urls_list.split(',')]
    
    # Use environment URLs if found
    if env_urls:
        urls = env_urls
    
    return urls


def main():
    parser = argparse.ArgumentParser(description='Test RTSP camera connections')
    parser.add_argument('--cameras', type=str, help='Comma-separated camera IDs to test (e.g., 0,1,2)')
    parser.add_argument('--url', type=str, help='Test a single RTSP URL')
    parser.add_argument('--methods', type=str, default='opencv,ffmpegcv', 
                       help='Test methods to use (opencv,ffmpegcv)')
    parser.add_argument('--parallel', action='store_true', help='Test cameras in parallel')
    parser.add_argument('--timeout', type=int, default=10, help='Connection timeout in seconds')
    
    args = parser.parse_args()
    
    # Parse methods
    methods = [m.strip() for m in args.methods.split(',')]
    
    # Get URLs to test
    if args.url:
        # Test single URL
        results = test_camera(0, args.url, methods)
        print(f"\nTest completed. Success: {results['overall_success']}")
        return 0 if results['overall_success'] else 1
    
    # Get configured URLs
    urls = get_rtsp_urls()
    if not urls:
        print("No RTSP URLs configured!")
        print("\nPlease set RTSP URLs using:")
        print("  - Environment variables: RTSP_URL_0, RTSP_URL_1, ...")
        print("  - Or comma-separated: RTSP_URLS=rtsp://cam1,rtsp://cam2")
        print("  - Or in config.py")
        return 1
    
    # Filter cameras if specified
    if args.cameras:
        camera_ids = [int(i.strip()) for i in args.cameras.split(',')]
        test_urls = [(i, urls[i]) for i in camera_ids if i < len(urls)]
    else:
        test_urls = list(enumerate(urls))
    
    print(f"Found {len(urls)} configured cameras, testing {len(test_urls)}")
    print(f"Test methods: {', '.join(methods)}")
    print(f"Parallel testing: {'Yes' if args.parallel else 'No'}")
    
    # Test cameras
    all_results = []
    start_time = time.time()
    
    if args.parallel:
        # Parallel testing
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(test_camera, cam_id, url, methods): (cam_id, url)
                for cam_id, url in test_urls
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    cam_id, url = futures[future]
                    print(f"Error testing camera {cam_id}: {e}")
    else:
        # Sequential testing
        for cam_id, url in test_urls:
            try:
                result = test_camera(cam_id, url, methods)
                all_results.append(result)
            except Exception as e:
                print(f"Error testing camera {cam_id}: {e}")
    
    # Summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get('overall_success', False))
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total cameras tested: {len(all_results)}")
    print(f"Successful connections: {successful}/{len(all_results)}")
    print(f"Failed connections: {len(all_results) - successful}")
    print(f"Total test time: {elapsed_time:.2f} seconds")
    
    if successful < len(all_results):
        print("\nFailed cameras:")
        for result in all_results:
            if not result.get('overall_success', False):
                print(f"  - Camera {result['camera_id']}: {result['url']}")
    
    # Save results to file
    results_file = f"rtsp_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"RTSP Camera Test Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")
        
        for result in all_results:
            f.write(f"Camera {result['camera_id']}: {result['url']}\n")
            f.write(f"Status: {'SUCCESS' if result['overall_success'] else 'FAILED'}\n")
            for method, details in result['tests'].items():
                f.write(f"  {method}: {'✓' if details['success'] else '✗'}\n")
                if details['success'] and 'details' in details:
                    for k, v in details['details'].items():
                        f.write(f"    - {k}: {v}\n")
                elif details.get('error'):
                    f.write(f"    - Error: {details['error']}\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if successful == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())