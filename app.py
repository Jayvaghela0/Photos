from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import torch
from PIL import Image
import numpy as np
import gdown
import requests

# Flask app initialize karein
app = Flask(__name__)
CORS(app)

# Temporary folders for uploaded and processed images
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Model folder and path
MODEL_FOLDER = "ESRGAN/models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "RRDB_ESRGAN_x4.pth")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Google Drive model URL
MODEL_URL = "https://drive.google.com/uc?id=1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN"

# Google Drive se model download karne ka function
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):  # Check if download successful
                raise Exception("Download failed using gdown")
        except Exception as e:
            print(f"gdown failed: {e}. Trying alternate method...")
            try:
                response = requests.get(MODEL_URL, stream=True)
                with open(MODEL_PATH, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print("Model downloaded successfully using requests!")
            except Exception as e:
                print(f"Alternate download method also failed: {e}")
                raise Exception("Model download failed. Please check the URL.")

# ESRGAN model load karne ka function
def load_esrgan_model():
    global model
    if 'model' not in globals():
        download_model()
        from ESRGAN.models.architectures import RRDBNet  # ESRGAN model class
        model = RRDBNet(3, 3, 64, 23)  # Model initialize karein
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)  # Load model with strict=False
        model.eval()
    return model

# Enhance image using ESRGAN
def enhance_image_using_esrgan(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype('float32') / 255.0  # Normalize to [0,1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW

    with torch.no_grad():
        enhanced_img = model(img).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

    enhanced_img = (enhanced_img * 255).astype('uint8')
    processed_image_path = os.path.join(PROCESSED_FOLDER, "enhanced_" + os.path.basename(image_path))
    Image.fromarray(enhanced_img).save(processed_image_path)

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

    # Load ESRGAN model (once globally)
    model = load_esrgan_model()

    # Enhance image
    processed_image_path = enhance_image_using_esrgan(file_path, model)

    # Return processed image
    return send_file(processed_image_path, mimetype='image/jpeg')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
