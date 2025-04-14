import argparse
import cv2
import numpy as np

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Overlay an image on a larger frame")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the larger frame image")
    parser.add_argument('--overlay_image', type=str, required=True, help="Path to the image to overlay")
    parser.add_argument('--output_image', type=str, required=True, help="Path to save the output image")
    return parser.parse_args()

# === Main Code ===
def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if cap.isOpened():
        success, frame = cap.read()

    cap.release()
    if frame is None:
        raise FileNotFoundError(f"File '{args.video_path}' not found.")

    overlay_image = cv2.imread(args.overlay_image)

    if overlay_image is None:
        raise FileNotFoundError(f"File '{args.overlay_image}' not found.")

    overlay_height, overlay_width = overlay_image.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    x_offset, y_offset = 100, 100 
    end_x = x_offset + overlay_width
    end_y = y_offset + overlay_height

    if end_x > frame_width or end_y > frame_height:
        scale_factor = min(frame_width / overlay_width, frame_height / overlay_height)
        overlay_image = cv2.resize(overlay_image, (int(overlay_width * scale_factor), int(overlay_height * scale_factor)))
        overlay_height, overlay_width = overlay_image.shape[:2]

        end_x = x_offset + overlay_width
        end_y = y_offset + overlay_height

    frame[y_offset:end_y, x_offset:end_x] = overlay_image

    cv2.rectangle(frame, (x_offset, y_offset), (end_x, end_y), (0, 255, 0), 2)

    cv2.imwrite(args.output_image, frame)
    print(f"Saved the output to '{args.output_image}'")

if __name__ == "__main__":
    main()
