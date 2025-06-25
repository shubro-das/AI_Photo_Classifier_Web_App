import numpy as np
from tensorflow.keras.models import load_model
from backend.human_detection import process_image

hair_classes = ["Curly", "Straight", "Wavy"]
face_classes = ["Neutral", "Smile"]
clothing_classes = ["Casual", "Formal"]

# Load models once
hair_model = load_model("backend/models/hairstyle_classifier.h5")
face_model = load_model("backend/models/facial_expression_classifier_resnet50.h5")
clothing_model = load_model("backend/models/clothing_classifier_resnet50_phase2.h5")

def predict_all(image_path):
    hair_input, face_input, clothing_input = process_image(image_path)

    hair_pred = hair_model.predict(hair_input)
    face_score = face_model.predict(face_input)[0][0]
    cloth_score = clothing_model.predict(clothing_input)[0][0]

    return {
        "Hairstyle": hair_classes[np.argmax(hair_pred)],
        "Facial Expression": face_classes[int(face_score > 0.5)],
        "Clothing Style": clothing_classes[int(cloth_score > 0.5)]
    }
