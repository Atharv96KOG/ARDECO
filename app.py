from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

# === Load SAM Model (ViT-H) ===
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure this file exists
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cpu")

predictor = SamPredictor(sam)

# === OpenCV Face Detector (to exclude face areas) ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Default Wall Color (Can be changed dynamically)
WALL_COLOR = (50, 120, 200)  # Light Blue (BGR Format)

# === Detect Faces (to avoid painting on them) ===
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)

    return face_mask

# === Segment the Wall using SAM ===
def get_wall_mask(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_points = np.array([
        [50, 50], [frame.shape[1] - 50, 50],
        [50, frame.shape[0] - 50], [frame.shape[1] - 50, frame.shape[0] - 50]
    ])
    input_labels = np.array([1, 1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask

# === Remove Small Unwanted Segments ===
def filter_small_masks(mask, min_area=20000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Ignore background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1
    return filtered_mask.astype(bool)

# === Apply Color Mask to the Wall ===
def apply_color_mask(frame, mask, color, alpha=0.7):
    colored_frame = frame.copy()
    colored_frame[mask] = (np.array(color) * alpha + colored_frame[mask] * (1 - alpha)).astype(np.uint8)
    return colored_frame

# === Video Streaming Function ===
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not detected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Detect and segment the wall
        wall_mask = get_wall_mask(frame)
        wall_mask = filter_small_masks(wall_mask)

        # Detect faces and exclude from the wall mask
        face_mask = detect_faces(frame)
        final_wall_mask = np.logical_and(wall_mask, ~face_mask)

        # Apply dynamic wall color
        painted_frame = apply_color_mask(frame, final_wall_mask, WALL_COLOR, alpha=0.7)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', painted_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color', methods=['POST'])
def set_color():
    global WALL_COLOR
    color = request.form.get('color', '#3278C8')  # Default to light blue
    WALL_COLOR = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))  # Convert Hex to BGR
    return "Color Updated", 200

if __name__ == '__main__':
    app.run(debug=True)
