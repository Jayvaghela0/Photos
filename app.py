from flask import Flask, request, send_file, jsonify
import os
import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGAN

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MODEL_PATH = "weights/RealESRGAN_x4plus.pth"  # Pre-trained model path
model = None  # Global model variable

def load_ai_model():
    global model
    if model is None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RealESRGAN(device, scale=4)
            model.load_weights(MODEL_PATH, download=True)
        except Exception as e:
            print(f"Error loading AI model: {e}")
            model = None

def enhance_image(image_path):
    if model is None:
        load_ai_model()
        if model is None:
            return None

    img = Image.open(image_path).convert('RGB')
    try:
        enhanced_img = model.predict(img)
    except Exception as e:
        print(f"Error during AI inference: {e}")
        return None

    processed_image_path = os.path.join(PROCESSED_FOLDER, "enhanced_" + os.path.basename(image_path))
    enhanced_img.save(processed_image_path)
    return processed_image_path

@app.route('/')
def home():
    return "AI Image Enhancer Backend is Running"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    processed_image_path = enhance_image(file_path)
    if processed_image_path is None:
        return jsonify({"error": "AI model processing failed"}), 500

    return send_file(processed_image_path, mimetype='image/jpeg', as_attachment=True, download_name=os.path.basename(processed_image_path))

if __name__ == '__main__':
    app.run(debug=True) 
