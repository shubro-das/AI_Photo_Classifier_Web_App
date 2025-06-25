# image_processing.py

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# Path to the Caffe deploy file and weights for person detection
PERSON_PROTOTXT = r"backend\model_files\deploy.prototxt"
PERSON_MODEL    = r"backend\model_files\mobilenet_iter_73000.caffemodel"

print("Using prototxt path:", PERSON_PROTOTXT)
print("Using model path:", PERSON_MODEL)


# Confidence threshold for the person detector
CONF_THRESHOLD = 0.5

# Target size for all three classifiers
TARGET_SIZE = (224, 224)

# ─── LOAD THE CAFFE PERSON DETECTOR ────────────────────────────────────────────

# Use OpenCV's DNN module to load the Caffe model
person_net = cv2.dnn.readNetFromCaffe(PERSON_PROTOTXT, PERSON_MODEL)


# ─── UTILITIES FOR CROPPING AND PREPROCESSING ──────────────────────────────────

def detect_primary_person(img: np.ndarray) -> tuple:
    """
    Run the Caffe model on the input image to detect all 'person' boxes.
    Returns the single bounding box (x, y, w, h) of the highest-confidence 'person'.
    If no person >= CONF_THRESHOLD is found, returns None.
    """
    (h, w) = img.shape[:2]
    # Create a blob: swapRB=True, size=300×300 (as per typical MobileNet‐SSD)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 scalefactor=1.0/127.5,
                                 size=(300, 300),
                                 mean=(127.5, 127.5, 127.5),
                                 swapRB=True,
                                 crop=False)
    person_net.setInput(blob)
    detections = person_net.forward()

    best_box = None
    best_conf = 0.0

    # Each detection ⟶ [batch_id, class_id, confidence, x1, y1, x2, y2]
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        # In the standard MobileNet-SSD proto, class 15 = "person"
        if class_id == 15 and confidence > CONF_THRESHOLD:
            # Extract box (relative coordinates → absolute)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2 - x1, y2 - y1)

    return best_box  # (x, y, width, height) or None


def compute_face_hair_crop(person_box: tuple, img_shape: tuple) -> tuple:
    """
    Given a person bounding box (x, y, w, h) and the full image shape (H, W, _),
    compute a square crop around the top of the person box that includes
    forehead, ears, hair, neck, and upper shoulders.

    Returns (x_start, y_start, x_end, y_end), clipped to image boundaries.
    """
    (x, y, w, h) = person_box
    (H, W) = img_shape[:2]

    # We want a square whose side length = w (width of person), 
    # anchored at the top of the person box, but if that overshoots,
    # we clamp within [0, H] vertically.

    side = w
    x_center = x + w // 2

    # Horizontal span: make it square of width=w, centered around the person box
    x1 = max(0, x_center - side // 2)
    x2 = x1 + side
    if x2 > W:
        # shift left if overflow
        x2 = W
        x1 = W - side

    # Vertical span: start at y (top of person box). y + side:
    y1 = y
    y2 = y1 + side
    if y2 > H:
        # If it exceeds bottom, clamp
        y2 = H
        y1 = max(0, H - side)

    return (int(x1), int(y1), int(x2), int(y2))


def crop_region(img: np.ndarray, rect: tuple) -> np.ndarray:
    """
    Given the full image (H×W×3) and a rectangle (x1, y1, x2, y2),
    return the cropped sub-image.
    """
    (x1, y1, x2, y2) = rect
    return img[y1:y2, x1:x2]


def preprocess_for_hairstyle(crop: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, convert to float in [0,1], expand dims → shape (1,224,224,3).
    """
    # Resize
    resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # Convert BGR → RGB (since Keras' load_img expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = keras_image.img_to_array(rgb)  # shape (224,224,3), float32
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)  # (1,224,224,3)


def preprocess_for_face_expression(crop: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, apply ResNet50 preprocess_input, expand dims → shape (1,224,224,3).
    """
    resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = keras_image.img_to_array(rgb)  # (224,224,3)
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)         # applies mean-subtraction, BGR→RGB swap, etc.
    return arr  # (1,224,224,3)


def preprocess_for_clothing(crop: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, apply ResNet50 preprocess_input, expand dims → shape (1,224,224,3).
    Exactly the same as face_expression preprocessing.
    """
    return preprocess_for_face_expression(crop)


# ─── MAIN ENTRY POINT ───────────────────────────────────────────────────────────

def process_image(image_path: str) -> dict:
    """
    Given the path to an input image (any resolution), this function:
     1. Reads the image via OpenCV.
     2. Detects the primary human bounding box.
     3. Computes two crops:
          a) Face+Hair region (square) 
          b) Full-body region (the person box itself)
     4. Applies the exact preprocessing for each downstream model:
          - hairstyle_model: resize→224×224, /255, expand dims
          - facial_expression_model: resize→224×224, ResNet50 preprocess_input
          - clothing_model: resize→224×224, ResNet50 preprocess_input
     5. Returns a dict with three NumPy arrays, each ready for model.predict().

    Return format:
      {
        "hair_input": np.ndarray (1,224,224,3),
        "face_input": np.ndarray (1,224,224,3),
        "clothing_input": np.ndarray (1,224,224,3)
      }
    If no person is detected (confidence < threshold), raises a RuntimeError.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Read image (OpenCV loads as BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 2. Detect primary person
    person_box = detect_primary_person(img)
    if person_box is None:
        raise RuntimeError("No person detected with confidence ≥ {:.2f}".format(CONF_THRESHOLD))

    (px, py, pw, ph) = person_box

    # 3a. Full-body crop = exactly the person bbox
    x1_body, y1_body = px, py
    x2_body, y2_body = px + pw, py + ph
    full_body_crop = crop_region(img, (x1_body, y1_body, x2_body, y2_body))

    # 3b. Hair crop
    hair_rect = compute_face_hair_crop(person_box, img.shape)
    hair_crop = crop_region(img, hair_rect)

    # 3c. Face crop only (tight face box using Haar cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        raise RuntimeError("No face detected for face crop!")

    # Select the most confident face (usually the largest area)
    face = max(faces, key=lambda rect: rect[2] * rect[3])
    (xf, yf, wf, hf) = face
    xf, yf = max(0, xf), max(0, yf)
    xf2, yf2 = xf + wf, yf + hf
    xf2, yf2 = min(img.shape[1], xf2), min(img.shape[0], yf2)

    face_crop = img[yf:yf2, xf:xf2]



    

    # 4. Preprocess for each model
    hair_input     = preprocess_for_hairstyle(hair_crop)
    face_input     = preprocess_for_face_expression(face_crop)
    clothing_input = preprocess_for_clothing(full_body_crop)

           # === Visualization of Cropped Regions on Original Image ===
    vis_img = img.copy()

    # Ensure coordinates are within bounds and integers
    x1_body, y1_body = max(0, int(x1_body)), max(0, int(y1_body))
    x2_body, y2_body = min(img.shape[1], int(x2_body)), min(img.shape[0], int(y2_body))

    fx1, fy1, fx2, fy2 = [int(v) for v in hair_rect]
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(img.shape[1], fx2), min(img.shape[0], fy2)

    print(f"[DEBUG] Body Box: ({x1_body}, {y1_body}) → ({x2_body}, {y2_body})")
    print(f"[DEBUG] Face+Hair Box: ({fx1}, {fy1}) → ({fx2}, {fy2})")

    # Draw bounding boxes
    cv2.rectangle(vis_img, (x1_body, y1_body), (x2_body, y2_body), (0, 255, 0), 2)
    cv2.putText(vis_img, "Body", (x1_body, max(0, y1_body - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(vis_img, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
    cv2.putText(vis_img, "Hair", (fx1, max(0, fy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.rectangle(vis_img, (xf, yf), (xf2, yf2), (0, 0, 255), 2)
    cv2.putText(vis_img, "Face", (xf, max(0, yf - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # Resize for fixed window
    display_width = 600
    scale = display_width / vis_img.shape[1]
    vis_img_resized = cv2.resize(vis_img, (display_width, int(vis_img.shape[0] * scale)))

    cv2.imshow("Cropped Regions Visualization", vis_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return hair_input, face_input, clothing_input


# ─── USAGE EXAMPLE ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python image_processing.py <path_to_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    try:
        outputs = process_image(input_path)
        print("Successfully processed image.")
        print("hair_input.shape    =", outputs["hair_input"].shape)
        print("face_input.shape    =", outputs["face_input"].shape)
        print("clothing_input.shape=", outputs["clothing_input"].shape)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
