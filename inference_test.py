import argparse
import cv2
import numpy as np
import json
import time  # For capturing current time
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm  # Import tqdm for progress bar


path = r'Videos/video.mp4'
model_object_detection_path = r'project_conveyer/Models/yolodet_fp16/best.engine'
model_image_segmantation_path = r'project_conveyer/Models/yoloseg_fp16/best_yoloseg.engine'
output_path = r'project_conveyer/inference_output'
tracker_config = r'project_conveyer/bytetrack_custom.yaml'
conf_threshold = 0.4
output_data_path = r'project_conveyer/output_data.json'
colors = [144, 80, 70]
area_segment_path = r'project_conveyer/area.json'
ratio_path = r'project_conveyer/ratio.json'
red_lines_coordinate = r'project_conveyer/red_lines_coordinate.json'

with open(area_segment_path, "r") as f:
    area_config = json.load(f)

with open(ratio_path, "r") as f:
    ratio_config = json.load(f)

with open(red_lines_coordinate, "r") as f:
    red_lines_coordinate = json.load(f)

cm_per_pixel = float(ratio_config["cm_per_pixel"])
area_for_segment = tuple(area_config)
red_lines_coordinate =tuple(red_lines_coordinate)

# === Load models ===
model = YOLO(model_object_detection_path, task='detect')
model_seg = YOLO(model_image_segmantation_path, task='segment')

# === Video input/output ===
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))