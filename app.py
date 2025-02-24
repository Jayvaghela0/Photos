from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # CORS ke liye
import os
import torch
from PIL import Image
import numpy as np
import logging  # Debugging ke liye

# Flask app initialize karein
app = Flask(__name__)
CORS(app)  # CORS enable karein

# Logging configure karein
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Temporary folders for uploaded and processed images
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Model file ka path
MODEL_FOLDER = "ESRGAN/models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "RRDB_PSNR_x4.pth")  # Manually added file

# Ensure models folder exists
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ESRGAN model load karein
def load_esrgan_model():
    try:
        from ESRGAN.models.architectures import RRDBNet  # ESRGAN model class
        model = RRDBNet(3, 3, 64, 23)  # Example parameters
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # CPU par load karein
        model.eval()  # Set model to evaluation mode
        logger.info("ESRGAN model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading ESRGAN model: {e}")
        raise

# Enhance image using ESRGAN
def enhance_image_using_esrgan(image_path, model):
    try:
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
        logger.info(f"Enhanced image saved: {processed_image_path}")
        return processed_image_path
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        raise

# Home route
@app.route('/')
def home():
    return "ESRGAN Image Enhancer Backend"

# Image upload and enhancement route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        logger.error("No image uploaded")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        # Save uploaded image
        file.save(file_path)
        logger.info(f"Image saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return jsonify({"error": "Error saving image"}), 500

    try:
        # Load ESRGAN model
        model = load_esrgan_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({"error": "Error loading model"}), 500

    try:
        # Enhance image using ESRGAN
        processed_image_path = enhance_image_using_esrgan(file_path, model)
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return jsonify({"error": "Error enhancing image"}), 500

    try:
        # Return processed image
        return send_file(processed_image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error sending file: {e}")
        return jsonify({"error": "Error sending file"}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
