from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS  # CORS ko import karein
import os
from ESRGAN.models import RRDBNet
from ESRGAN.utils import load_model, enhance_image
import torch

# Flask app initialize karein
app = Flask(__name__)
CORS(app)  # CORS ko enable karein

# Configure upload and enhanced image folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ENHANCED_FOLDER'] = 'static/enhanced_images'

# Load ESRGAN Model
model_path = "ESRGAN/models/RRDB_PSNR_x4.pth"  # Path to the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model = load_model(model, model_path, device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']
        if file.filename == '':
            return "No file selected!"
        if file:
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)

            # Enhance image using ESRGAN
            output_path = os.path.join(app.config['ENHANCED_FOLDER'], f"enhanced_{file.filename}")
            enhance_image(model, upload_path, output_path, device)

            return render_template('index.html', original_image=upload_path, enhanced_image=output_path)
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)
    app.run(debug=True)
