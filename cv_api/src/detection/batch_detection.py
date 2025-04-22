# detection.py (updated)
from ultralytics import YOLO
import numpy as np
import time
import cv2

class PotatoDetector:
    def __init__(self, model_path, warmup_image, tracker_config=None, segmentation_area=None):
        self.model = YOLO(model_path, task='detect')
        self.tracker_config = tracker_config
        self.segmentation_area = segmentation_area or []
        self.warmup(warmup_image)
        
    def warmup(self, warmup_image, warmup_frames=1, img_size=640):
        # Create dummy frames
        dummy_frame = cv2.imread(warmup_image)
        dummy_frame = dummy_frame[:, :, ::-1]
        
        # Warmup GPU by running inference multiple times
        print(f"Warming up Detection model with {warmup_frames} dummy frames...")
        start_time = time.time()
        
        for _ in range(warmup_frames):
            # Run both predict and track to warm up all operations
            _ = self.model.predict(dummy_frame, imgsz=img_size, verbose=True)
            _ = self.model.track(dummy_frame, imgsz=img_size, verbose=True, persist=False)
        
        warmup_time = time.time() - start_time
        print(f"Warmup completed in {warmup_time:.2f} seconds")

    def _in_segmentation_area(self, box_coords):
        if not self.segmentation_area:
            return True
            
        sa = self.segmentation_area
        return (box_coords[0] >= sa[0] and 
                box_coords[1] >= sa[1] and
                box_coords[2] <= sa[2] and
                box_coords[3] <= sa[3])

    def batch_track(self, frames):
        results = []
        with self.model.track(frames,
                           tracker=self.tracker_config,
                           verbose=False) as batch_results:
            
            for r in batch_results:
                frame_result = {'boxes': [], 'processing_time': r.speed['total']}
                
                if r.boxes:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        if self._in_segmentation_area(coords):
                            frame_result['boxes'].append({
                                'coords': coords,
                                'id': int(box.id.item()) if box.id else -1,
                                'conf': box.conf.item()
                            })
                results.append(frame_result)
        return results