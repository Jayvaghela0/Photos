from flask import Flask, request, jsonify, send_file
import torch
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load ESRGAN model
def load_esrgan_model(model_path):
    from ESRGAN import ESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    model_path = model_path

    # Initialize ESRGAN upscaler
    upscaler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    return upscaler

# Load the pre-trained model
model_path = "RRDB_PSNR_x4.pth"
upscaler = load_esrgan_model(model_path)

# Image enhancement function
def enhance_image(image):
    # Convert image to numpy array
    img = np.array(image)
    # Enhance image using ESRGAN
    output, _ = upscaler.enhance(img, outscale=4)
    # Convert back to PIL Image
    enhanced_image = Image.fromarray(output)
    return enhanced_image

# Flask route for image upload and enhancement
@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the uploaded image
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    # Enhance the image
    enhanced_image = enhance_image(image)

    # Save enhanced image to a temporary file
    output_buffer = io.BytesIO()
    enhanced_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    # Return the enhanced image
    return send_file(output_buffer, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
