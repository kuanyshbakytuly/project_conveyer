import argparse
import cv2
import numpy as np
import json
import time  # For capturing current time
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm  # Import tqdm for progress bar

# === Command-Line Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Track and segment potatoes in video")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the video file")
    parser.add_argument('--model_object_detection', type=str, required=True, help="Path to the object detection model")
    parser.add_argument('--model_image_segmentation', type=str, required=True, help="Path to the image segmentation model")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output video")
    parser.add_argument('--area_segment_path', type=str, default="area_config.json", help="Path to the configuration JSON file")
    parser.add_argument('--ratio_path', type=str, default="area_config.json", help="Path to the configuration JSON file")
    parser.add_argument('--tracker_config', type=str, default="bytetrack_custom.yaml", help="Path to the tracker configuration YAML file")
    parser.add_argument('--conf_threshold', type=float, default=0.4, help="Confidence threshold for detection")
    parser.add_argument('--output_data_path', type=str, required=True, help="Confidence threshold for detection")
    return parser.parse_args()


# Function to recursively convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.generic):  # This will check if the object is a numpy type
        return obj.item()  # Convert it to a native Python type
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}  # Recursively process dictionaries
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]  # Recursively process lists
    return obj

# === Main Code ===
def main():
    args = parse_args()

    camera_id = 1

    # === Configuration ===
    path = args.video_path
    model_object_detection_path = args.model_object_detection
    model_image_segmantation_path = args.model_image_segmentation
    output_path = args.output_path
    tracker_config = args.tracker_config
    conf_threshold = args.conf_threshold
    output_data_path = args.output_data_path
    colors = [144, 80, 70]

    with open(args.area_segment_path, "r") as f:
        area_config = json.load(f)

    with open(args.ratio_path, "r") as f:
        ratio_config = json.load(f)

    cm_per_pixel = float(ratio_config["cm_per_pixel"])
    area_for_segment = tuple(area_config)

    # === Load models ===
    model = YOLO(model_object_detection_path)
    model_seg = YOLO(model_image_segmantation_path)

    # === Video input/output ===
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # === Tracking state ===
    frame_count = 0
    counter = 0
    map_track_size = {}
    track_set = []

    # Prepare a dictionary to store all potato data
    potato_data = {}

    # Use tqdm to add a progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                tracker=tracker_config,
                verbose=False
            )

            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (area_for_segment[0], area_for_segment[1]),
                        (area_for_segment[2], area_for_segment[3]), (255, 0, 0), 2)

            potato_boxes = []
            potato_img_boxes = []

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    track_id = int(box.id.cpu().numpy()[0])
                    track_set.append(track_id)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    prev_size = map_track_size.get(track_id, (0, 0))
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"{round(prev_size[0], 2)}cm {round(prev_size[1], 2)}cm",
                                (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    if (area_for_segment[0] <= x1 <= area_for_segment[2] and
                        area_for_segment[1] <= y1 <= area_for_segment[3] and
                        area_for_segment[0] <= x2 <= area_for_segment[2] and
                        area_for_segment[1] <= y2 <= area_for_segment[3]):

                        img_box = frame[y1:y2, x1:x2]
                        potato_img_boxes.append(img_box)
                        potato_boxes.append([x1, y1, x2, y2, track_id])
                    
                    counter += 1

                    data = {
                        "camera_id": camera_id,
                        "potato_id": track_id,  # Unique track ID
                        "type": "potato",
                        "size": f"{round(prev_size[0], 2)}cm {round(prev_size[1], 2)}cm",
                        "coordinates": [x1, y1, x2, y2],
                        "frame_id": frame_count,  # Frame ID
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "passed": "yes",  # Example status
                        "sorted": "no"  # Example status
                    }

                    # Store the data in the dictionary using the potato_id as the key
                    potato_data[track_id] = data

            # Segmentation
            if potato_img_boxes:
                results_seg = model_seg.predict(potato_img_boxes, verbose=False)
                for i, result in enumerate(results_seg):
                    x1, y1, x2, y2, track_id = potato_boxes[i]
                    prev_major, prev_minor = map_track_size.get(track_id, (0, 0))

                    for mask in result.masks.xy:
                        abs_coords = mask + np.array([x1, y1])
                        abs_coords = abs_coords.astype(np.int32)

                        contour = np.int32([abs_coords]).reshape((-1, 1, 2))

                        if contour.shape[0] >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            _, axes, _ = ellipse
                            major_axis = max(axes) * cm_per_pixel
                            minor_axis = min(axes) * cm_per_pixel

                            avg_major = (major_axis + prev_major) / 2 if prev_major else major_axis
                            avg_minor = (minor_axis + prev_minor) / 2 if prev_minor else minor_axis

                            map_track_size[track_id] = (avg_major, avg_minor)

                            cv2.fillPoly(annotated_frame, [contour], colors)

            cv2.putText(annotated_frame, f"Potato Count: {max(track_set)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(annotated_frame)
            frame_count += 1

            pbar.update(1)  # Update the progress bar

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")
    cv2.destroyAllWindows()

    # Save the potato data to a JSON file after processing
    with open(output_data_path, 'w') as json_file:
        json.dump(convert_numpy_types(potato_data), json_file, indent=4)

if __name__ == '__main__':
    main()
