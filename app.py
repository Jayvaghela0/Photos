from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS import karein
import cv2
import numpy as np
import torch
from model import RRDBNet_arch
from utils import imwrite
import base64

app = Flask(__name__)
CORS(app)  # CORS enable karein

# Load ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet_arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = output.transpose(1, 2, 0)
    output = (output * 255.0).round()

    # Enhanced image ko base64 format mein convert karein
    _, img_encoded = cv2.imencode('.png', output)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'enhancedImage': f'data:image/png;base64,{img_base64}'})

if __name__ == '__main__':
    app.run(debug=True)
