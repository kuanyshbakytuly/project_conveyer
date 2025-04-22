# segmentation.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

class PotatoSegmentation:
    def __init__(self, model_path, warmup_image, ratio=0.01):
        self.model = YOLO(model_path, task='segment')
        self.ratio = ratio
        self.warmup(warmup_image)

    def warmup(self, warmup_image, warmup_frames: int = 1, img_size: int = 320):
        """Warm up the segmentation model."""
        print(f"Warming up Segmentation model with {warmup_frames} dummy frames...")
        start_time = time.time()
        dummy_frame = cv2.imread(warmup_image)
        dummy_frame = dummy_frame[:, :, ::-1]
        for _ in range(warmup_frames):
            _ = self.model.predict(dummy_frame, imgsz=img_size, verbose=True)
        warmup_time = time.time() - start_time
        print(f"Warmup completed in {warmup_time:.2f} seconds")

    def process_batch(self, frames, detections):
        batch_input = [self._prepare_input(f, d) for f, d in zip(frames, detections)]
        results = self.model.predict(batch_input, verbose=False)
        
        output = []
        for frame_idx, r in enumerate(results):
            frame_result = {
                'masks': [],
                'processing_time': r.speed['preprocess'] + r.speed['inference']
            }
            
            if r.masks is not None:
                for mask in r.masks:
                    contour = self._process_mask(mask.xy[0])
                    frame_result['masks'].append({
                        'contour': contour,
                        'size': self._calculate_size(contour)
                    })
            output.append(frame_result)
        return output

    def _prepare_input(self, frame, detection):
        # Crop ROIs based on detections
        if not detection['boxes']:
            return np.zeros((320,320,3), dtype=np.uint8)
        
        # Select largest detection
        boxes = [b['coords'] for b in detection['boxes']]
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        largest_idx = np.argmax(areas)
        x1,y1,x2,y2 = boxes[largest_idx].astype(int)
        return frame[y1:y2, x1:x2]

    def _process_mask(self, mask):
        contour = mask.astype(np.int32).reshape((-1,1,2))
        return contour

    def _calculate_size(self, contour):
        if len(contour) < 5:
            return (0, 0)
        ellipse = cv2.fitEllipse(contour)
        return (ellipse[1][0]*self.ratio, ellipse[1][1]*self.ratio)