from flask import Flask, request, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from ESRGAN.test import enhance_image  # ESRGAN ka function import karein

app = Flask(__name__)
CORS(app)  # CORS enable karein

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
ENHANCED_FOLDER = 'enhanced'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER

# ESRGAN model path
MODEL_PATH = 'ESRGAN/models/RRDB_PSNR_x4.pth'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files['image']
    if file.filename == '':
        return {"error": "No image selected"}, 400

    # Save uploaded image
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # Enhance image using ESRGAN
    output_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
    enhance_image(input_path, output_path, MODEL_PATH)

    # Return enhanced image
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
