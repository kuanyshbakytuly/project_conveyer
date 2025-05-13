import torch
import numpy as np
import time
import cv2
import queue
import threading
from collections import defaultdict
import ffmpegcv
import matplotlib.pyplot as plt
from ultralytics import YOLO

class BatchInferenceSystem:
    """System for batch processing of multiple video streams through detection and segmentation models"""
    
    def __init__(self, 
                 detection_model_path, 
                 segmentation_model_path,
                 detection_device=0,
                 segmentation_device=1,
                 batch_size=2,
                 segmentation_area=None,
                 cm_per_pixel=0.04,
                 frame_shape=(720, 1280, 3)):
        """
        Initialize the batch inference system.
        
        Args:
            detection_model_path (str): Path to detection model
            segmentation_model_path (str): Path to segmentation model
            detection_device (int): GPU device ID for detection
            segmentation_device (int): GPU device ID for segmentation
            batch_size (int): Maximum batch size for inference
            segmentation_area (list): [x1, y1, x2, y2] area to focus on for segmentation
            cm_per_pixel (float): Ratio for converting pixels to cm
            frame_shape (tuple): Expected frame shape (height, width, channels)
        """
        self.detection_model_path = detection_model_path
        self.segmentation_model_path = segmentation_model_path
        self.detection_device = detection_device
        self.segmentation_device = segmentation_device
        self.batch_size = batch_size
        self.segmentation_area = segmentation_area or [0, 0, frame_shape[1], frame_shape[0]]
        self.cm_per_pixel = cm_per_pixel
        self.frame_shape = frame_shape
        
        # Queues for the pipeline
        self.frame_queue = queue.Queue(maxsize=batch_size*2)
        self.detection_queue = queue.Queue(maxsize=batch_size)
        self.output_queue = queue.Queue(maxsize=batch_size)
        
        # Metrics and tracking state
        self.stream_metrics = defaultdict(list)
        self.stream_fps_list = []
        self.stream_frame_counts = defaultdict(int)
        self.stream_start_times = {}
        self.total_frames_processed = 0
        
        # Thread control
        self.stop_signal = threading.Event()
        
        # Color definitions for visualization
        self.COLORS = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 255)
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load detection and segmentation models"""
        print("Loading detection model...")
        self.detection_model = YOLO(self.detection_model_path, task='detect')
        
        print("Loading segmentation model...")
        self.segmentation_model = YOLO(self.segmentation_model_path, task='segment')

        print("Load")
        
        print("Models loaded successfully")
    
    def warmup_models(self, warmup_image=None):
        """
        Warm up models with dummy inference
        
        Args:
            warmup_image (str): Path to image for warmup, creates dummy if None
        """
        # Create dummy image if not provided
        if warmup_image:
            dummy_frame = cv2.imread(warmup_image)
            if dummy_frame is None:
                print(f"Warning: Could not read warmup image {warmup_image}")
                dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        else:
            dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        
        # Convert to RGB if needed (YOLO expects RGB)
        if len(dummy_frame.shape) == 3 and dummy_frame.shape[2] == 3:
            dummy_frame = dummy_frame[:, :, ::-1]  # BGR to RGB
        
        # Create a batch of dummy frames
        dummy_batch = [dummy_frame] * min(4, self.batch_size)  # Small batch for warmup
        
        # Warmup detection model
        print(f"Warming up detection model...")
        start_time = time.time()
        with torch.no_grad():
            _ = self.detection_model.predict(dummy_batch, verbose=False, device=self.detection_device)
            # Warmup with track operation (commented out as requested)
            # _ = self.detection_model.track(dummy_batch, verbose=False, persist=False, device=self.detection_device)
        print(f"Detection model warmup completed in {time.time() - start_time:.2f} seconds")
        
        # Warmup segmentation model
        print(f"Warming up segmentation model...")
        start_time = time.time()
        with torch.no_grad():
            _ = self.segmentation_model.predict(dummy_batch, verbose=False, device=self.segmentation_device)
        print(f"Segmentation model warmup completed in {time.time() - start_time:.2f} seconds")
    
    def frame_reader(self, stream_path, stream_id):
        """
        Thread function to read frames from a video stream
        
        Args:
            stream_path (str): Path to video file or stream
            stream_id (int): Unique identifier for this stream
        """
        try:
            # Initialize video capture
            cap = ffmpegcv.VideoCaptureNV(stream_path, pix_fmt='bgr24', 
                                         resize=(self.frame_shape[1], self.frame_shape[0]), 
                                         gpu=1)
            self.stream_start_times[stream_id] = time.time()
            
            while not self.stop_signal.is_set():
                frame_time = time.time()
                ret, frame = cap.read()
                frame_time = time.time() - frame_time
                if not ret:
                    print(f"End of stream {stream_id}")
                    break
                    
                frame_metrics = {
                    'frame_read_time': frame_time,
                    'stream_id': stream_id,
                    'frame_id': self.stream_frame_counts[stream_id],
                    'frame_read': 0.0,
                    'detection': 0.0,
                    'detection_processing': 0.0,
                    'segmentation': 0.0,
                    'drawing': 0.0,
                    'total': 0.0
                }
                
                # Put frame and metadata into queue
                self.frame_queue.put((frame.copy(), frame_metrics, stream_id))
                self.stream_frame_counts[stream_id] += 1
                
        except Exception as e:
            print(f"Error in frame reader for stream {stream_id}: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            print(f"Frame reader for stream {stream_id} finished")
    
    def detection_processor(self):
        """Thread function to process batches of frames for object detection"""
        try:
            batch_frames = []
            batch_metrics = []
            batch_stream_ids = []
            
            while not self.stop_signal.is_set():
                try:
                    frame, metrics, stream_id = self.frame_queue.get(timeout=1.0)
                
                    
                    # Add to batch
                    batch_frames.append(frame)
                    batch_metrics.append(metrics)
                    batch_stream_ids.append(stream_id)
                    
                    # Process when batch is full or queue is emptying
                    if len(batch_frames) >= self.batch_size or self.frame_queue.qsize() == 0:
                        if batch_frames:
                            # Process batch with detection model
                            print(f'Len(batch_frames) = {len(batch_frames)}')
                            detection_results = self._process_detection_batch(
                                batch_frames, batch_metrics, batch_stream_ids
                            )
                            
                            # Put results in next queue for segmentation
                            for result in detection_results:
                                self.detection_queue.put(result)
                            
                            # Reset batch
                            batch_frames = []
                            batch_metrics = []
                            batch_stream_ids = []
                    
                    self.frame_queue.task_done()
                    
                except queue.Empty:
                    if batch_frames:
                        # Process remaining frames in the batch
                        continue
                    # Check if all streams are done
                    if self._all_streams_finished():
                        break
                        
        except Exception as e:
            print(f"Error in detection processor: {str(e)}")
        finally:
            print("Detection processor finished")
    
    def _process_detection_batch(self, frames, metrics_list, stream_ids):
        """
        Process a batch of frames through the detection model
        
        Args:
            frames (list): List of frames to process
            metrics_list (list): List of metrics dictionaries
            stream_ids (list): List of stream IDs
            
        Returns:
            list: Detection results for each frame with metadata
        """
        # Run batch inference
        start_detection = time.time()
        with torch.no_grad():
            batch_results = self.detection_model.predict(
                frames, 
                verbose=False, 
                imgsz=640,
                device=self.detection_device
            )
        detection_time = time.time() - start_detection
        
        # Process results for each frame
        processed_results = []
        start_process = time.time()
        for i, (result, metrics, stream_id) in enumerate(zip(batch_results, metrics_list, stream_ids)):
            
            # Get original frame and convert back to BGR for drawing
            orig_frame = frames[i][:, :, ::-1] if frames[i].shape[2] == 3 else frames[i]
            
            # Extract detection metadata for tracking
            detection_metadata = self._extract_detection_metadata(
                result, stream_id, metrics['frame_id']
            )
            
            # Extract potential potato images from frame for segmentation
            potato_images, potato_boxes = self._extract_potato_crops(
                orig_frame, result, self.segmentation_area
            )
            
            # Add metrics
            metrics['detection'] = detection_time
            metrics['detection_processing'] = time.time() - start_process
            
            # Add to results
            processed_results.append((
                orig_frame,          # Original frame (BGR)
                potato_images,       # Cropped images for segmentation
                potato_boxes,        # Bounding boxes with metadata
                detection_metadata,  # Detection metadata for tracking
                metrics,             # Performance metrics
                stream_id            # Stream ID
            ))
        
        return processed_results
    
    def _extract_detection_metadata(self, result, stream_id, frame_id):
        """
        Extract detection metadata for tracking
        
        Args:
            result: YOLO detection result
            stream_id: Stream identifier
            frame_id: Frame number
            
        Returns:
            dict: Detection metadata
        """
        metadata = {
            'stream_id': stream_id,
            'frame_id': frame_id,
            'boxes': [],
            'confidences': [],
            'classes': []
        }
        
        if result.boxes is None or len(result.boxes) == 0:
            return metadata
            
        # Extract boxes, confidences, and classes
        boxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
        confs = result.boxes.conf.cpu().numpy().tolist() if hasattr(result.boxes, 'conf') else [0.0] * len(boxes)
        cls_idx = result.boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(result.boxes, 'cls') else [0] * len(boxes)
        
        metadata['boxes'] = boxes
        metadata['confidences'] = confs
        metadata['classes'] = cls_idx
        
        return metadata
    
    def _extract_potato_crops(self, frame, result, detection_area):
        """
        Extract potato crops from frame based on detection results
        
        Args:
            frame: Original frame (BGR)
            result: YOLO detection result
            detection_area: [x1, y1, x2, y2] area for filtering detections
            
        Returns:
            tuple: (list of cropped images, list of bounding boxes with metadata)
        """
        potato_images = []
        potato_boxes = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return potato_images, potato_boxes
            
        # Get bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        
        # Check if boxes' centers are in detection area
        if len(detection_area) == 4:
            da = np.array(detection_area)
            # Calculate box centers
            center_x = (boxes[:, 0] + boxes[:, 2]) / 2
            center_y = (boxes[:, 1] + boxes[:, 3]) / 2
            # Check if centers are within detection area
            in_da = (
                (center_x >= da[0]) & 
                (center_x <= da[2]) & 
                (center_y >= da[1]) & 
                (center_y <= da[3])
            )
            boxes = boxes[in_da]
            
        # Extract crops and metadata
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Make sure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Only add if we have a valid box
            if x2 > x1 and y2 > y1:
                # Create crop (RGB version for segmentation model)
                crop = frame[y1:y2, x1:x2]
                crop_rgb = crop[:, :, ::-1]  # BGR to RGB
                potato_images.append(crop_rgb)
                
                # Add box with index for later mapping
                # Format: [x1, y1, x2, y2, box_index]
                potato_boxes.append([x1, y1, x2, y2, i])
        
        return potato_images, potato_boxes
    
    def segmentation_processor(self):
        """Thread function to process batches of frames for segmentation"""
        try:
            batch_data = []
            
            while not self.stop_signal.is_set():
                try:
                    frame, potato_images, potato_boxes, det_metadata, metrics, stream_id = self.detection_queue.get(timeout=1.0)
                    
                    # Store for batch processing
                    batch_data.append((frame, potato_images, potato_boxes, det_metadata, metrics, stream_id))
                    
                    # Process when batch is full or queue is emptying
                    if len(batch_data) >= self.batch_size or self.detection_queue.qsize() == 0:
                        print(f'len(seg_batch) = {len(batch_data)}')
                        if batch_data:
                            # Process batch with segmentation model
                            results = self._process_segmentation_batch(batch_data)
                            
                            # Put results in output queue
                            for result in results:
                                self.output_queue.put(result)
                            
                            # Reset batch
                            batch_data = []
                    
                    self.detection_queue.task_done()
                    
                except queue.Empty:
                    if batch_data:
                        # Process remaining items in the batch
                        continue
                    # Check if all streams are done
                    if self._all_streams_finished() and self.detection_queue.empty():
                        break
                        
        except Exception as e:
            print(f"Error in segmentation processor: {str(e)}")
        finally:
            print("Segmentation processor finished")
    
    def _process_segmentation_batch(self, batch_data):
        """
        Process a batch of potato images with the segmentation model
        
        Args:
            batch_data: List of (frame, potato_images, potato_boxes, det_metadata, metrics, stream_id)
            
        Returns:
            list: Results with segmentation data added
        """
        results = []
        
        # Collect all potato images from all frames into a single batch
        all_potato_images = []
        potato_frame_indices = []  # Map each potato back to its frame index
        
        for frame_idx, (_, potato_images, _, _, _, _) in enumerate(batch_data):
            all_potato_images.extend(potato_images)
            potato_frame_indices.extend([frame_idx] * len(potato_images))
        
        # Skip segmentation if no potato images
        if not all_potato_images:
            # Just pass through the original data with empty segmentation results
            for frame, _, potato_boxes, det_metadata, metrics, stream_id in batch_data:
                metrics['segmentation'] = 0.0
                metrics['total'] = sum([
                    metrics['frame_read'],
                    metrics['detection'],
                    metrics['detection_processing'],
                    metrics['segmentation'],
                    metrics['drawing']
                ])
                results.append((frame, det_metadata, None, metrics, stream_id))
            return results
        
        # Run segmentation on all potato images at once
        start_segmentation = time.time()
        print(f'len(all_potato_images) = {len(all_potato_images)}')
        with torch.no_grad():
            seg_results = self.segmentation_model.predict(
                all_potato_images,
                imgsz=160,
                verbose=False,
                device=self.segmentation_device
            )
        seg_time = time.time() - start_segmentation
        
        # Reorganize segmentation results by frame
        frame_to_segmentations = defaultdict(list)
        for i, (seg_result, frame_idx) in enumerate(zip(seg_results, potato_frame_indices)):
            frame_to_segmentations[frame_idx].append((i, seg_result))
        
        # Process each frame's data
        for frame_idx, (frame, potato_images, potato_boxes, det_metadata, metrics, stream_id) in enumerate(batch_data):
            # Get segmentation results for this frame
            frame_segmentations = frame_to_segmentations.get(frame_idx, [])
            
            # Measure processing time
            start_process = time.time()
            
            # Calculate sizes and process segmentation results
            potato_sizes = []
            
            for seg_idx, (potato_idx, seg_result) in enumerate(frame_segmentations):
                if seg_idx < len(potato_boxes):
                    box = potato_boxes[seg_idx]
                    # Calculate size from mask if available
                    if seg_result.masks is not None:
                        width, height = self._calculate_size_from_mask(
                            seg_result.masks.data, 
                            self.cm_per_pixel
                        )
                        potato_sizes.append((box[4], width, height))  # (box_index, width, height)
            
            # Draw results on frame (only if it's for visualization)
            processed_frame = self._draw_results(
                frame, 
                det_metadata,
                potato_boxes,
                potato_sizes
            )
            
            # Update metrics
            metrics['segmentation'] = seg_time
            metrics['drawing'] = time.time() - start_process
            metrics['total'] = sum([
                metrics['frame_read'],
                metrics['detection'],
                metrics['detection_processing'],
                metrics['segmentation'],
                metrics['drawing']
            ])
            
            # Add segmentation data to detection metadata
            seg_data = {
                'potato_boxes': potato_boxes,
                'potato_sizes': potato_sizes
            }
            
            # Store result
            results.append((processed_frame, det_metadata, seg_data, metrics, stream_id))
        
        return results
    
    def _calculate_size_from_mask(self, mask_data, ratio):
        """
        Calculate width and height from segmentation mask
        
        Args:
            mask_data: Mask data from YOLOv8 segmentation
            ratio: Pixel to cm ratio
            
        Returns:
            tuple: (width, height) in centimeters
        """
        if mask_data is None or len(mask_data) == 0:
            return 0, 0
            
        # Convert mask tensor to numpy array
        mask = mask_data[0].cpu().numpy()
        
        # Find non-zero elements in mask
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            return 0, 0
        
        # Calculate width and height
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        width = (max_x - min_x) * ratio
        height = (max_y - min_y) * ratio
        
        return width, height
    
    def _draw_results(self, frame, det_metadata, potato_boxes, potato_sizes):
        """
        Draw detection and segmentation results on the frame
        
        Args:
            frame: Original frame
            det_metadata: Detection metadata
            potato_boxes: List of potato boxes [x1, y1, x2, y2, box_idx]
            potato_sizes: List of (box_idx, width, height) tuples
            
        Returns:
            frame: Processed frame with visualizations
        """
        # Create a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Draw detection boxes
        for box in det_metadata['boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), self.COLORS['red'], 2)
        
        # Draw detection area if specified
        if self.segmentation_area and len(self.segmentation_area) == 4:
            x1, y1, x2, y2 = self.segmentation_area
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), self.COLORS['blue'], 2)
            
        # Draw sizes for potatoes that were segmented
        for box_idx, width, height in potato_sizes:
            # Find the corresponding box
            for box in potato_boxes:
                if box[4] == box_idx:
                    x1, y1, x2, y2 = box[:4]
                    cv2.putText(
                        result_frame, 
                        f"Size: {width:.2f}cm x {height:.2f}cm",
                        (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, 
                        self.COLORS['green'], 
                        2
                    )
                    break
        
        return result_frame
    
    def results_processor(self):
        """Thread function to handle results and maintain metrics"""
        try:
            while not self.stop_signal.is_set():
                try:
                    frame, det_metadata, seg_data, metrics, stream_id = self.output_queue.get(timeout=1.0)
                    
                    # Store metrics
                    self.stream_metrics[stream_id].append(metrics)
                    self.total_frames_processed += 1
                    
                    # This is where you would store detection metadata for tracking
                    # (Left as a placeholder for your custom tracker implementation)
                    
                    # For example:
                    # if your_custom_tracker is not None:
                    #     your_custom_tracker.update(det_metadata, seg_data)
                    
                    self.output_queue.task_done()
                    
                except queue.Empty:
                    if self._all_streams_finished() and self.output_queue.empty():
                        break
                        
        except Exception as e:
            print(f"Error in results processor: {str(e)}")
        finally:
            # Calculate stream FPS
            for stream_id in self.stream_metrics:
                end_time = time.time()
                total_time = end_time - self.stream_start_times.get(stream_id, end_time)
                fps = self.stream_frame_counts[stream_id] / total_time if total_time > 0 else 0
                self.stream_fps_list.append(fps)
                print(f"Stream {stream_id}: {self.stream_frame_counts[stream_id]} frames in {total_time:.2f}s = {fps:.2f} FPS")
                
            print("Results processor finished")
    
    def _all_streams_finished(self):
        """Check if all stream reader threads have completed"""
        return all(not thread.is_alive() for thread in self.reader_threads)
    
    def generate_performance_report(self):
        """Generate performance analysis report and plots"""
        all_fps = []
        time_stats = {
            'frame_read': 0.0,
            'detection': 0.0,
            'detection_processing': 0.0,
            'segmentation': 0.0,
            'drawing': 0.0
        }

        total_frames = 0
        for stream_id, metrics in self.stream_metrics.items():
            for frame in metrics:
                total_frames += 1
                
                if frame['total'] > 0:
                    all_fps.append(1 / frame['total'])
                else:
                    all_fps.append(0)
                
                for key in time_stats:
                    time_stats[key] += frame.get(key, 0)

        avg_fps = np.mean(all_fps) if all_fps else 0
        avg_stream_fps = np.mean(self.stream_fps_list) if self.stream_fps_list else 0

        total_time = sum(time_stats.values())
        time_percentages = {k: v/total_time*100 for k, v in time_stats.items()} if total_time > 0 else {}

        stream_fps_text = "\n".join([f"Stream {i}: {fps:.2f} FPS" for i, fps in enumerate(self.stream_fps_list)])
        
        summary_text = f"""Performance Summary:
        -------------------
        Average End-to-End Stream FPS: {avg_stream_fps:.2f}
        Individual Stream FPS:
        {stream_fps_text}

        Average Per-Frame FPS: {avg_fps:.2f}

        Time Distribution (% of total processing time):
        - Frame Reading: {time_percentages.get('frame_read', 0):.1f}%
        - Detection (Model): {time_percentages.get('detection', 0):.1f}%
        - Detection (Processing): {time_percentages.get('detection_processing', 0):.1f}%
        - Segmentation (Model): {time_percentages.get('segmentation', 0):.1f}%
        - Drawing Results: {time_percentages.get('drawing', 0):.1f}%

        Absolute Times per Frame (ms):
        - Frame Reading: {time_stats['frame_read']/total_frames*1000:.2f}
        - Detection Model: {time_stats['detection']/total_frames*1000:.2f}
        - Detection Processing: {time_stats['detection_processing']/total_frames*1000:.2f}
        - Segmentation Model: {time_stats['segmentation']/total_frames*1000:.2f}
        - Drawing Results: {time_stats['drawing']/total_frames*1000:.2f}
        """

        print(summary_text)
        
        # Create performance visualization plots
        plt.figure(figsize=(20, 15))
        
        # Stream FPS plot
        plt.subplot(3, 3, 7)
        plt.bar(range(len(self.stream_fps_list)), self.stream_fps_list, color='cyan')
        plt.title('End-to-End FPS per Stream')
        plt.xlabel('Stream ID')
        plt.ylabel('FPS')
        plt.grid(True)
        plt.xticks(range(len(self.stream_fps_list)))
        
        # Summary text
        plt.subplot(3, 3, 1)
        plt.axis('off')
        plt.text(0, 0.5, summary_text, fontfamily='monospace', fontsize=9)

        # FPS plot
        plt.subplot(3, 3, 2)
        plt.plot(all_fps, color='blue')
        plt.title('Frame Rate Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('FPS')
        plt.grid(True)
        plt.ylim(0, max(all_fps)*1.2 if all_fps else 30)

        # Time distribution pie chart
        plt.subplot(3, 3, 4)
        labels = [
            'Frame Read', 
            'Detection Model', 
            'Detection Processing',
            'Segmentation Model',
            'Drawing Results'
        ]
        sizes = [time_percentages.get(k, 0) for k in time_stats]
        explode = (0.1, 0, 0, 0, 0)
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=140)
        plt.title('Processing Time Distribution (%)')
        plt.axis('equal')

        # Detailed timing plot (ms)
        plt.subplot(3, 3, 5)
        times_ms = [time_stats[k]/total_frames*1000 for k in time_stats]
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, times_ms, align='center', alpha=0.7)
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel('Time (ms)')
        plt.title('Average Time per Processing Stage')
        plt.grid(True)

        # Cumulative time plot
        plt.subplot(3, 3, 6)
        cumulative_times = np.cumsum(times_ms)
        plt.plot(cumulative_times, marker='o', color='purple')
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel('Cumulative Time (ms)')
        plt.title('Cumulative Processing Time')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('batch_inference_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Performance plots saved as 'batch_inference_performance.png'")
        
        return summary_text
    
    def run(self, video_paths):
        """
        Run the batch inference system on multiple video streams
        
        Args:
            video_paths (list): List of paths to video files
            
        Returns:
            str: Performance summary text
        """
        # Store thread references
        self.reader_threads = []
        
        # Start frame reader threads (one per stream)
        for stream_id, video_path in enumerate(video_paths):
            thread = threading.Thread(
                target=self.frame_reader,
                args=(video_path, stream_id)
            )
            self.reader_threads.append(thread)
            thread.start()
        
        # Start processor threads
        detection_thread = threading.Thread(target=self.detection_processor)
        segmentation_thread = threading.Thread(target=self.segmentation_processor)
        results_thread = threading.Thread(target=self.results_processor)
        
        detection_thread.start()
        segmentation_thread.start()
        results_thread.start()
        
        # Wait for all frame readers to complete
        for thread in self.reader_threads:
            thread.join()
        
        # Wait for processors to finish remaining work
        detection_thread.join()
        segmentation_thread.join()
        results_thread.join()
        
        # Generate performance report
        return self.generate_performance_report()
    
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    
    inference = BatchInferenceSystem(
                detection_model_path = 'Models/model_det_int8_0_20/best.engine', 
                 segmentation_model_path = 'Models/model_seg_fp16_0_20/best_yoloseg.engine',
                 detection_device=0,
                 segmentation_device=0,
                 batch_size=20,
                 segmentation_area=[630, 0, 650, 720],
                 cm_per_pixel=0.02997,
                 frame_shape=(720, 1280, 3))
    video_paths = ['video.mov']*20
    inference.run(video_paths)