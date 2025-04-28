import ffmpegcv
import multiprocessing as mp
import torch
from pynvml import *
import cv2
import psutil
import os
import time

def process_stream(stream, stream_id, output_queue):
    try:
        import pycuda.autoinit
        torch.cuda.init()
        nvmlInit()

        test_tensor = torch.randn(10, device='cuda')
        del test_tensor


        cap = ffmpegcv.VideoCaptureNV(stream, pix_fmt='bgr24', resize=(640, 640))
        print('Inited cap')
        frame_id = 0
        FPS_LIMIT = 25  # или 10
        frame_interval = 1.0 / FPS_LIMIT

        while True:
            start_time = time.time()
            ret, orig_frame = cap.read()

            if not ret:
                break

            with torch.no_grad():
                frame = orig_frame.copy()

                frame = cv2.resize(frame, (640, 360))
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                _, jpeg = cv2.imencode('.jpg', frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            
                # Send to output queue if connected
                try:
                    output_queue.put(jpeg.tobytes(), timeout=0)
                except mp.queues.Full:
                    print('Queue is full')

                #run model
                #send output frame
            frame_id += 1

            if frame_id % 30==0:
                print(f'{stream_id}: 1 sec')


    except Exception as e:
        print(f"Process {stream_id} error: {str(e)}")
    finally:
        nvmlShutdown()