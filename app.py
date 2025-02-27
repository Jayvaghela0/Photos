from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from ESRGAN.test import enhance_image  # Import the enhance_image function

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create upload and output folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle image upload and enhancement
@app.route("/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Enhance the image using ESRGAN
        output_filename = f"enhanced_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        enhance_image(input_path, output_path)

        # Return the enhanced image URL
        return jsonify({"enhanced_image_url": f"/output/{output_filename}"}), 200

    return jsonify({"error": "Invalid file type"}), 400

# Route to serve enhanced images
@app.route("/output/<filename>")
def output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# Health check route
@app.route("/")
def health_check():
    return "Image Enhancer is running!"

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
