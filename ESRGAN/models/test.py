import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np

# Load ESRGAN model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img.width // 4 * 4, img.height // 4 * 4), Image.BICUBIC)  # Resize for model
    img_tensor = ToTensor()(img).unsqueeze(0)
    return img_tensor

# Postprocess image
def postprocess_image(output_tensor, output_path):
    output_img = ToPILImage()(output_tensor.squeeze(0).clamp(0, 1))
    output_img.save(output_path)

# Enhance image
def enhance_image(input_path, output_path, model_path):
    # Load model
    model = load_model(model_path)

    # Preprocess image
    img_tensor = preprocess_image(input_path)

    # Enhance image
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # Postprocess and save image
    postprocess_image(enhanced_tensor, output_path)
