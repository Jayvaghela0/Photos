from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # CORS इनेबल कर दिया गया है

# मॉडल डाउनलोड करने का सिस्टम
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = "weights/RealESRGAN_x4plus.pth"

os.makedirs("weights", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading RealESRGAN model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# Model लोड करें
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True
)

@app.route('/sharpen', methods=['POST'])
def sharpen_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)

    try:
        output, _ = upscaler.enhance(img, outscale=4)
        img_pil = Image.fromarray(output)
        img_io = io.BytesIO()
        img_pil.save(img_io, format="JPEG")
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return jsonify({"image": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
