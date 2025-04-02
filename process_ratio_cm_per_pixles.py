import argparse
import cv2
import numpy as np
import json

drawing = False
start_point = (-1, -1)
lines = []

# Function to draw overlay text on the image
def draw_overlay_text(img):
    text = "Mark up Ruler in frame | Press 'c' to clear, 'Esc' to exit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = 10
    text_y = img.shape[0] - 10
    cv2.putText(img, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

# Mouse callback function to draw lines
def draw_line(event, x, y, flags, param):
    global drawing, start_point, lines, frame  # Declare frame as global

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

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Mark up ruler in the frame and save config")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_json', type=str, required=True, help="Path to the output JSON file")
    return parser.parse_args()

# === Main Code ===
def main():
    global frame  # Make frame global to access it inside draw_line function
    args = parse_args()

    # === Load the frame ===
    frame = cv2.imread(args.input_image)
    if frame is None:
        raise FileNotFoundError(f"File '{args.input_image}' not found.")
    seg_fram = frame.copy()
    cv2.imwrite('frame.jpg', seg_fram)

    frame = draw_overlay_text(frame)
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", draw_line)

    # === Main loop ===
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        elif key == ord('c'):  # Clear
            lines.clear()
            frame = seg_fram.copy()
            frame = draw_overlay_text(frame)
            cv2.imshow("Frame", frame)

    # === Calculate boundaries and ruler scale ===
    if lines:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for x1, y1, x2, y2 in lines:
            min_x = min(min_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
            print(f"Start: ({x1}, {y1}), End: ({x2}, {y2})")

        area_for_segment = [int(min_x), int(min_y), int(max_x), int(max_y)]
        cm_per_pixel = 10 / (max_x - min_x)  # Assuming the line is 10 cm in real life

        print(f"area_for_segment: {area_for_segment}")
        print(f"cm_per_pixel: {cm_per_pixel}")

        # === Save the data to JSON file ===
        with open(args.output_json, "w") as f:
            json.dump({
                "cm_per_pixel": cm_per_pixel
            }, f, indent=2)
        print(f"Saved to '{args.output_json}'")

    else:
        print("Lines not marked.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
