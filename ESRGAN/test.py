import torch
from .utils.image_utils import load_image, save_image
from .utils.model_utils import load_model

def enhance_image(input_path, output_path, model_path="ESRGAN/models/RRDB_PSNR_x4.pth"):
    """
    Enhance an image using the ESRGAN model.
    """
    # Load the model
    model = load_model(model_path)

    # Load the input image
    img = load_image(input_path)

    # Enhance the image
    with torch.no_grad():
        enhanced_img = model(img)

    # Save the enhanced image
    save_image(enhanced_img, output_path)
