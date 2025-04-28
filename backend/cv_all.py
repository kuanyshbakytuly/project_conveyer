import ffmpegcv
import multiprocessing as mp
import torch
from pynvml import *
import cv2
import tensorrt as trt
import psutil
import os
from cv_api.src.detection.detection import PotatoDetector
from cv_api.src.segmentation.segmentation import PotatoSegmentation
import time

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def process_stream(stream, stream_id, output_queue):
    try:
        import pycuda.autoinit
        torch.cuda.init()
        nvmlInit()

        test_tensor = torch.randn(10, device='cuda')
        del test_tensor
        # Initialization code
        # Original values for 3840×2160:
        # segmentation_area = [1295, 1, 2338, 2154]  # [x1, y1, x2, y2]
        # cm_per_pixel = 0.00999000999000999

        # Scaled values for 1920×1080:
        # segmentation_area = [648, 0, 1169, 1080]  # All coordinates divided by 2
        # cm_per_pixel = 0.01998  # Since pixels are now larger (half resolution), each pixel covers more cm

        segmentation_area = [432, 0, 779, 720]  # [x1, y1, x2, y2]
        cm_per_pixel = 0.02997
        conf = 0.6
        tracker_config = 'bytetrack_custom.yaml'

        # Initialize models
        print(f'Initting Detector --- process {stream_id} ')
        detection = PotatoDetector(
            model_path='Models/model_det_test/best.engine',
            camera_id=stream_id,
            task='detect',
            tracker_config=tracker_config,
            segmentation_area=segmentation_area,
            track_ids=set(),
            warmup_image='frame.jpg'
        )
        print(f'Initting Segmenter ---process {stream_id} ')
        segmentation = PotatoSegmentation(
            model_path='Models/model_seg_fp16/best_yoloseg.engine',
            ratio=cm_per_pixel,
            warmup_image='box.jpg'
        )
        print(f'Initting VideoCapture --- process {stream_id} ')
        cap = ffmpegcv.VideoCaptureNV(stream, pix_fmt='bgr24', resize=(1280, 720))
        #out = ffmpegcv.VideoWriterNV(output_stream, 'h264', 25)

        frame_id = 0

        while True:    
            ret, orig_frame = cap.read()

            if not ret:
                break

            with torch.no_grad():
                inp = orig_frame.copy()

                # Detection
                track_result = detection.track(frame=inp, frame_id=frame_id, conf=conf)
                potato_images = track_result[0]
                potato_boxes = track_result[1]

                # Segmentation
                if potato_images:
                    detection.tracked_sizes, inp = segmentation.process_batch(
                        potato_images,
                        potato_boxes,
                        detection.tracked_sizes,
                        inp
                    )
                
                inp = cv2.resize(inp, (448, 252))


                if frame_id != 0:
                    quality = 70
                    queue_size = output_queue.qsize()

                    
                    if queue_size > 3:
                        quality = max(30, quality - 10)
                    elif queue_size < 2:
                        quality = min(90, quality + 5)

                    _, jpeg = cv2.imencode('.jpg', inp, 
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                
                    try:
                        output_queue.put(jpeg.tobytes(), timeout=0.01)
                    except mp.queues.Full:
                        print('Queue is full')
                        pass  # Skip frame if queue is full
            frame_id += 1


    except Exception as e:
        print(f"Process {stream_id} error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'detection' in locals():
            del detection
        if 'segmentation' in locals():
            del segmentation
        nvmlShutdown()
        print('End')