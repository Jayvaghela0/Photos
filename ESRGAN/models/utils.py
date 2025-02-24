import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def enhance_image(model, input_path, output_path, device):
    img = Image.open(input_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output_img = ToPILImage()(output.squeeze(0).cpu())
    output_img.save(output_path)
