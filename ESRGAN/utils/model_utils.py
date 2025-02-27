import torch

def load_model(model_path):
    """
    Load the ESRGAN model.
    """
    model = torch.load(model_path, map_location=torch.device("cpu"))  # Use CPU for compatibility
    model.eval()  # Set model to evaluation mode
    return model
