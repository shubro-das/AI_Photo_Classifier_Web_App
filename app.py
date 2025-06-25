import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import random
from backend.predictor import predict_all  

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and returns predictions."""
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400

    if file and allowed_file(file.filename):
        # Check file size (your HTML requires > 50KB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0) # Reset file pointer to the beginning

        # if file_size < 50 * 1024: # 50 KB in bytes
        #     return jsonify({"error": "Image size is less than 50KB. Please upload a larger image for better results."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform prediction 
        predictions = predict_all(filepath)

        # You might want to delete the file after processing, or store it
        # os.remove(filepath)

        return jsonify(predictions), 200
    else:
        return jsonify({"error": "Allowed image types are png, jpg, jpeg, gif"}), 400

if __name__ == '__main__':
    app.run(debug=True)










