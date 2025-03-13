import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# === Load SAM model (CPU) ===
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cpu")

predictor = SamPredictor(sam)

# === Helper function to apply color mask ===
def apply_color_mask(frame, mask, color=(255, 200, 200), alpha=0.7):
    colored_frame = frame.copy()
    colored_frame[mask] = (np.array(color) * alpha + colored_frame[mask] * (1 - alpha)).astype(np.uint8)
    return colored_frame

# === Function to detect face (OpenCV Haar Cascade) ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)  # Draw white box over face
    return face_mask

# === Function to segment wall area ===
def get_wall_mask(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Place points near edges where walls are likely to appear (away from center/face)
    input_points = np.array([
        [50, 50], [frame.shape[1] - 50, 50],  # Top-left & Top-right corners
        [50, frame.shape[0] - 50], [frame.shape[1] - 50, frame.shape[0] - 50]  # Bottom-left & Bottom-right corners
    ])
    input_labels = np.array([1, 1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    # Combine all masks into one
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask

# === Filter small unwanted masks (like face-sized objects) ===
def filter_small_masks(mask, min_area=20000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Ignore background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1
    return filtered_mask.astype(bool)

# === Webcam Processing ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected.")
    exit()

WALL_COLOR = (50, 120, 200)  # Light pinkish wall color (BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Step 1: Detect wall mask
    wall_mask = get_wall_mask(frame)

    # Step 2: Remove small unwanted segments (like face-sized blobs)
    wall_mask = filter_small_masks(wall_mask)

    # Step 3: Optional - Detect and mask face (so it won't be colored)
    face_mask = detect_face_mask(frame)

    # Final wall mask = Wall mask minus face region
    final_wall_mask = np.logical_and(wall_mask, ~face_mask)

    # Step 4: Apply wall color
    painted_frame = apply_color_mask(frame, final_wall_mask, color=WALL_COLOR, alpha=0.7)

    # Display result
    cv2.imshow("Real-Time Wall Painting", painted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
