# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y wget

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Real-ESRGAN model weights
RUN mkdir -p weights && wget -O weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth

# Copy app files
COPY . .

# Expose port and run app
EXPOSE 5000
CMD ["python", "app.py"]
