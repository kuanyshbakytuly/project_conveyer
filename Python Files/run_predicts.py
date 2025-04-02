import cv2
import numpy as np
from multiprocessing import Pool
from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("best.pt")  # or choose yolov8m/l-world.pt

import glob
videos = sorted(glob.glob('vides/*'))

def process_video(model, video_path, output_video_path):
    """
    Process a single video and save the output with predictions.
    
    Parameters:
    - model: Trained model for prediction.
    - video_path: Path to the input video.
    - output_video_path: Path to save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the model
        #processed_frame = preprocess_frame(frame)

        # Predict using the model
        results = model.predict(frame, conf=0.45)

        # Draw the prediction on the frame
        frame_with_prediction = draw_prediction_on_frame(frame, results)

        # Write the processed frame to the output video
        out.write(frame_with_prediction)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

def preprocess_frame(frame):
    """
    Preprocess the frame for model prediction.
    """
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame_normalized, axis=0)

def draw_prediction_on_frame(frame, results):
    """
    Draw the prediction result on the frame.
    """
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
        label = box.cls[0]  # Get class label (if available)
        conf = box.conf[0]  # Confidence score
        conf_rounded = np.round(conf.numpy(), 2) if hasattr(conf, 'numpy') else np.round(conf, 2)
    
        # Convert confidence to string for OpenCV to display
        conf_str = str(conf_rounded)

        # Draw the bounding box on the image
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, conf_str, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def process_multiple_videos_in_parallel(model, video_paths, output_video_paths):
    """
    Process multiple videos simultaneously using multiprocessing.
    
    Parameters:
    - model: Trained model for prediction.
    - video_paths: List of paths to the input videos.
    - output_video_paths: List of paths to save the output videos.
    """
    # Create a pool of processes for parallel video processing
    with Pool(processes=len(video_paths)) as pool:
        # Map the function to the video paths and output paths
        pool.starmap(process_video, zip([model] * len(video_paths), video_paths, output_video_paths))

video_paths = videos[20:]

output_video_paths = ['Yolov_model1_output/'+i.split('/')[-1] for i in video_paths]

if __name__ == '__main__':
    process_multiple_videos_in_parallel(model, video_paths, output_video_paths)
