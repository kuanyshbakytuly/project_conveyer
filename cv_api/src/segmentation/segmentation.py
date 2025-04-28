import time
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO

class PotatoSegmentation:
    """Handles potato segmentation and size measurement using YOLO segmentation model."""
    
    def __init__(self, model_path: str, ratio: float = 0.1, warmup_image=None):
        """
        Initialize the segmentation processor.
        
        Args:
            model_path (str): Path to segmentation model weights
            cm_per_pixel (float): Conversion factor from pixels to centimeters
        """
        self.model = YOLO(model_path, task='segment')
        #self.warmup(warmup_image, warmup_frames=1)
        self.ratio = ratio
        self.phase_times = {
            'model_seg': [],
            'postprocessing': [],
            'size_calculation': []
        }
        
    def warmup(self, warmup_image, warmup_frames: int = 5, img_size: int = 320):
        """Warm up the segmentation model."""
        print(f"Warming up Segmentation model with {warmup_frames} dummy frames...")
        start_time = time.time()
        dummy_frame = cv2.imread(warmup_image)
        dummy_frame = dummy_frame[:, :, ::-1]
        for _ in range(warmup_frames):
            _ = self.model.predict(dummy_frame, imgsz=img_size, verbose=True)
        warmup_time = time.time() - start_time
        print(f"Warmup completed in {warmup_time:.2f} seconds")

    def process_batch(
        self,
        potato_images: List[np.ndarray],
        potato_boxes: List[List[int]],
        tracked_sizes: Dict[int, Tuple[float, float]],
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> Tuple[Dict[int, Tuple[float, float]], np.ndarray]:
        """
        Process a batch of potato images through segmentation pipeline.
        
        Args:
            potato_images: List of cropped potato images
            potato_boxes: List of [x1,y1,x2,y2,track_id] for each potato
            tracked_sizes: Dictionary of previous size measurements
            frame: Frame to draw annotations on
            color: Color for segmentation masks
            
        Returns:
            Updated tracked_sizes and annotated frame
        """
        if not potato_images:
            return tracked_sizes, frame

        # Run segmentation
        start_seg = time.time()
        results_seg = self.model.predict(potato_images, imgsz=320, verbose=False)
        self.phase_times['model_seg'].append(time.time() - start_seg)

        # Process results
        annotated_frame = frame.copy()
        
        for i, (result, box) in enumerate(zip(results_seg, potato_boxes)):
            x1, y1, x2, y2, track_id = box
            prev_major, prev_minor = tracked_sizes.get(track_id, (0, 0))
            
            if result.masks:
                self._process_mask(
                    result.masks.xy,
                    (x1, y1),
                    tracked_sizes,
                    track_id,
                    prev_major,
                    prev_minor,
                    annotated_frame,
                    color
                )
        
        return tracked_sizes, annotated_frame

    def _process_mask(
        self,
        masks: List[np.ndarray],
        offset: Tuple[int, int],
        tracked_sizes: Dict[int, Tuple[float, float]],
        track_id: int,
        prev_major: float,
        prev_minor: float,
        frame: np.ndarray,
        color: Tuple[int, int, int]
    ) -> None:
        """Process individual segmentation mask and calculate sizes."""
        for mask in masks:
            # Convert to absolute coordinates
            abs_coords = mask + np.array(offset)
            contour = abs_coords.astype(np.int32).reshape((-1, 1, 2))
            
            if len(contour) >= 5:
                start_calc = time.time()
                major, minor = self._calculate_axes_lengths(contour)
                
                # Update with running average
                avg_major = (major + prev_major) / 2 if prev_major else major
                avg_minor = (minor + prev_minor) / 2 if prev_minor else minor
                tracked_sizes[track_id] = (avg_major, avg_minor)
                
                self.phase_times['size_calculation'].append(time.time() - start_calc)
                cv2.fillPoly(frame, [contour], color)

    def _calculate_axes_lengths(self, contour: np.ndarray) -> Tuple[float, float]:
        """Calculate major and minor axis lengths from contour."""
        ellipse = cv2.fitEllipse(contour)
        _, axes, _ = ellipse
        return max(axes) * self.ratio, min(axes) * self.ratio
    
    def __del__(self):
        del self.model
