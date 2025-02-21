from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the reference logo
logo = cv2.imread("logo.png", 0)  # Convert to grayscale

def remove_logo(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Match template (detect logo)
    result = cv2.matchTemplate(gray, logo, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = logo.shape

    # Create a mask for inpainting
    mask = np.zeros_like(gray)
    mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = 255

    # Inpaint the image (remove logo)
    output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    output_path = "output.png"
    cv2.imwrite(output_path, output)
    return output_path
@app.route('/')
def home():
    return "Welcome to Photos."
@app.route('/remove_logo', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = "input.png"
    file.save(file_path)

    output_path = remove_logo(file_path)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
