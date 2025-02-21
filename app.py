from flask import Flask, request, send_file, jsonify
import os
import torch
from PIL import Image
import numpy as np
from ESRGAN.test import enhance_image  # ESRGAN ka helper function

app = Flask(__name__)

# Temporary folders for uploaded and processed images
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load ESRGAN model
def load_esrgan_model():
    model_path = "ESRGAN/models/RRDB_ESRGAN_x4.pth"  # Pretrained model ka path
    model = torch.load(model_path, map_location=torch.device('cpu'))  # CPU par load karein
    return model

# Enhance image using ESRGAN
def enhance_image(image_path):
    # Load ESRGAN model
    model = load_esrgan_model()

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]

    # Apply ESRGAN model
    enhanced_img = enhance_image(model, img)  # ESRGAN ka helper function use karein
    enhanced_img = (enhanced_img * 255).astype('uint8')  # Convert back to [0, 255]

    # Save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, "enhanced_" + os.path.basename(image_path))
    enhanced_img = Image.fromarray(enhanced_img)
    enhanced_img.save(processed_image_path)

    return processed_image_path

# Home route
@app.route('/')
def home():
    return "ESRGAN Image Enhancer Backend"

# Image upload and enhancement route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Enhance image using ESRGAN
    processed_image_path = enhance_image(file_path)

    # Return processed image
    return send_file(processed_image_path, mimetype='image/jpeg')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
