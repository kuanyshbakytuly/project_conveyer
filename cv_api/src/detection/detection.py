from ultralytics import YOLO
import cv2
import numpy as np
import time

class PotatoDetector:
    """A class for detecting and tracking potatoes in video frames using YOLO."""
    
    def __init__(self, model_path, camera_id=0, task='detect', tracker_config=None, segmentation_area=None, track_ids=None, warmup_image=None):
        """
        Initialize the potato detector.
        
        Args:
            model_path (str): Path to the YOLO model weights
            camera_id (int): Camera identifier
            task (str): YOLO task type ('detect', 'segment', etc.)
            detection_area (list): Coordinates of detection area [x1, y1, x2, y2]
            track_ids (set): Set of tracked potato IDs
        """
        self.model = YOLO(model_path, task=task)
        self.warmup(warmup_image)
        self.camera_id = camera_id
        self.tracker_config = tracker_config
        self.segmentation_area = segmentation_area or []
        self.tracked_ids = track_ids or set()
        self.tracked_sizes = {}
        self.potato_data = []
        
        # Color definitions
        self.COLORS = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 255)
        }

    def warmup(self, warmup_image, warmup_frames=10, img_size=640):
        """
        Warm up the model by running inference on dummy frames.
        
        Args:
            warmup_frames (int): Number of warmup frames to process
            img_size (int): Size of the dummy frames
        """
        
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

    def predict(self, frame, conf=0.6, img_size=640):
        """
        Run YOLO prediction on a frame.
        """
        return self.model.predict(frame, imgsz=img_size, conf=conf)

    def track(self, frame, frame_id=0, conf=0.6, img_size=None):
        """
        Track objects in a frame and process results.
        """
        start_tracker = time.time()
        if img_size is None:
            self.result_tracking = self.model.track(
                frame, conf=conf, tracker=self.tracker_config
            )
        else:
            self.result_tracking = self.model.track(
                frame, imgsz=img_size, conf=conf, tracker=self.tracker_config
            )
        tracker_time = time.time() - start_tracker

        # Measure processing and drawing times
        start_process = time.time()
        potato_images, potato_boxes, drawing_time = self._process_tracking_results(frame, frame_id)
        process_total_time = time.time() - start_process
        processing_time = process_total_time - drawing_time

        return (potato_images, potato_boxes, tracker_time, processing_time, drawing_time)
    
    def _process_tracking_results(self, frame, frame_id):
        """
        Fully vectorized processing of tracking results
        """
        if not self.result_tracking or self.result_tracking[0].boxes is None:
            return [], []

        boxes = self.result_tracking[0].boxes
        
        if boxes.id is None:
            return [], []

        all_coords = boxes.xyxy.cpu().numpy().astype(int)
        all_ids = boxes.id.cpu().numpy().astype(int)
        
        current_sizes = np.zeros((len(all_ids), 2))
        for i, tid in enumerate(all_ids):
            current_sizes[i] = self.tracked_sizes.get(tid, (0, 0))
        
        da = np.array(self.segmentation_area[:4])
        in_da = ((all_coords[:, 0] >= da[0]) &
                (all_coords[:, 1] >= da[1]) &
                (all_coords[:, 2] <= da[2]) &
                (all_coords[:, 3] <= da[3]))
        
        process_mask = in_da & (current_sizes == 0).all(axis=1)
        
        clean_frame = frame.copy()
        if np.any(process_mask):
            crop_coords = all_coords[process_mask]
            
            potato_images = []
            for x1, y1, x2, y2 in crop_coords:
                potato_images.append(clean_frame[y1:y2, x1:x2])
            
            potato_boxes = np.column_stack((
                crop_coords,
                all_ids[process_mask]
            )).tolist()
        else:
            potato_images = []
            potato_boxes = []
            
        drawing_time = 0.0
        for i in range(len(all_coords)):
            x1, y1, x2, y2 = all_coords[i]
            track_id = all_ids[i]
            height, width = current_sizes[i]
            
            self.tracked_ids.add(track_id)
            start_draw = time.time()
            self._store_potato_data(
                track_id, x1, y1, x2, y2, 
                height, width, frame_id
            )
            
            self._draw_box(frame, [x1, y1, x2, y2], 'red')
            self._draw_id_and_size(
               frame, x1, y1, track_id, 
               height, width
            )
            drawing_time += time.time() - start_draw
        
        return potato_images, potato_boxes, drawing_time


    def _store_potato_data(self, track_id, x1, y1, x2, y2, width, height, frame_id):
        """
        Store potato detection data.
        """
        data = {
            "camera_id": self.camera_id,
            "potato_id": track_id,
            "type": "potato",
            "size": f"{round(width, 2)}cm {round(height, 2)}cm",
            "coordinates": [x1, y1, x2, y2],
            "frame_id": frame_id,
            "passed": "yes",
            "sorted": "no"
        }
        
        while len(self.potato_data) <= track_id:
            self.potato_data.append(None)
        self.potato_data[track_id] = data

    def _draw_box(self, frame, coordinates, color_name):
        """Draw a bounding box on the frame."""
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS[color_name], 2)

    def _draw_id_and_size(self, frame, x, y, track_id, width, height):
        """Draw ID and size text on the frame."""
    
        cv2.putText(
            frame, f"ID: {track_id} {round(width, 2)}cm {round(height, 2)} cm ", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS['white'], 2
        )

    def draw_lines(self):
        """Draw reference lines (to be implemented)."""
        pass

    def __del__(self):
        del self.model
        