import argparse
import cv2
import numpy as np
import json
import os

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Mark up lines on a frame and save the area as a JSON file")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input frame image")
    parser.add_argument('--output_json', type=str, required=True, help="Path to save the output JSON file")
    return parser.parse_args()

# Initialize variables
drawing = False
start_point = (-1, -1)
lines = []

# === Main Code ===
def main():
    global frame  # Make frame global to access it inside draw_line function
    args = parse_args()

    # Load frame from video
    cap = cv2.VideoCapture(args.video_path)
    if cap.isOpened():
        success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise FileNotFoundError(f"Image '{args.video_path}' not found.")
    
    seg_fram = frame.copy()

    # Function to add overlay text to the frame
    def draw_overlay_text(img):
        overlay_text = "Mark up lines in frame | Press 'c' to clear, 'Esc' to exit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        size = cv2.getTextSize(overlay_text, font, scale, thickness)[0]
        text_x = 10
        text_y = img.shape[0] - 10  # Place at bottom
        cv2.putText(img, overlay_text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img

    # Mouse callback to draw lines
    def draw_line(event, x, y, flags, param):
        global drawing, start_point, lines, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            frame_copy = frame.copy()
            cv2.line(frame_copy, start_point, (x, y), (255, 0, 0), 2)
            frame_copy = draw_overlay_text(frame_copy)
            cv2.imshow("Frame", frame_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            lines.append((start_point[0], start_point[1], end_point[0], end_point[1]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            frame = draw_overlay_text(frame)
            cv2.imshow("Frame", frame)

    frame = draw_overlay_text(frame)
    cv2.imshow("Frame", frame)

    cv2.setMouseCallback("Frame", draw_line)

    # Interaction loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('c'):  # 'c' to clear
            lines.clear()
            show_frame = seg_fram.copy()
            cv2.imshow("Frame", show_frame)

    # Calculate area based on drawn lines
    if lines:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for x1, y1, x2, y2 in lines:
            min_x = min(min_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)

        area_for_segment = [int(min_x), int(min_y), int(max_x), int(max_y)]
        print("area_for_segment =", area_for_segment)

        # Save as .json
        with open(args.output_json, "w") as f:
            json.dump(area_for_segment, f, indent=2)
        print(f"Saved to {args.output_json}")

    else:
        print("No lines drawn. Area not saved.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
