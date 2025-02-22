from flask import Flask, request, send_file, jsonify
import os
import torch
from PIL import Image
import numpy as np
import gdown  # Google Drive se model download karne ke liye

app = Flask(__name__)

# Temporary folders for uploaded and processed images
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Model file ka path
MODEL_FOLDER = "ESRGAN/models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "RRDB_ESRGAN_x4.pth")

# Ensure models folder exists
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Google Drive se model download karein (sirf ek baar)
def download_model():
    if not os.path.exists(MODEL_PATH):  # Check if file already exists
        model_url = "https://drive.google.com/uc?id=1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN"  # Google Drive file ID
        print("Downloading model from Google Drive...")
        gdown.download(model_url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model already exists. Skipping download.")

# ESRGAN model load karein
def load_esrgan_model():
    download_model()  # Ensure model is downloaded
    from ESRGAN.models.architectures import RRDBNet  # ESRGAN model class
    model = RRDBNet(3, 3, 64, 23)  # Example parameters
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # CPU par load karein
    model.eval()  # Set model to evaluation mode
    return model

# Enhance image using ESRGAN
def enhance_image_using_esrgan(image_path, model):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]

    # Convert image to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW

    # Apply ESRGAN model
    with torch.no_grad():  # Disable gradient calculation
        enhanced_img = model(img).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()  # BCHW to HWC

    # Convert back to [0, 255]
    enhanced_img = (enhanced_img * 255).astype('uint8')

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

    # Load ESRGAN model
    model = load_esrgan_model()

    # Enhance image using ESRGAN
    processed_image_path = enhance_image_using_esrgan(file_path, model)

    # Return processed image
    return send_file(processed_image_path, mimetype='image/jpeg')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
