import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from ESRGAN.RRDBNet_arch import RRDBNet  # RRDBNet_arch se RRDBNet import karein

# Load ESRGAN model
def load_model(model_path):
    model = RRDBNet(3, 3, 64, 23, gc=32)  # RRDBNet model define karein
    model.load_state_dict(torch.load(model_path), strict=True)
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
