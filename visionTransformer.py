import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import requests
import os


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Vision Transformer model
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.eval()
model.to(device)
print("Model loaded successfully")

# Image preprocessing
preprocess = weights.transforms()

# Load an image from URL
def load_image_from_url(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        print("Image downloaded and preprocessed successfully")
        return image
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

# Predict the class of an image
def predict(image_url):
    image = load_image_from_url(image_url)
    if image is None:
        return None
    image = image.to(device)
    try:
        with torch.no_grad():
            outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        print("Prediction completed successfully")
        return probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Download and load class names
def download_and_load_labels():
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_path = "imagenet_classes.txt"
    try:
        # Download labels file if not exists
        if not os.path.exists(labels_path):
            response = requests.get(labels_url)
            with open(labels_path, "w") as f:
                f.write(response.text)
            print("Labels file downloaded successfully")
        
        # Load labels from file
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print("Labels loaded successfully")
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# Main function
def main():
    image_url = r"https://images.unsplash.com/photo-1533450718592-29d45635f0a9?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8anBnfGVufDB8fDB8fHww"
    labels = download_and_load_labels()
    if not labels:
        print("Failed to load labels. Exiting.")
        exit()

    probabilities = predict(image_url)
    if probabilities is None:
        print("Failed to predict. Exiting.")
        exit()

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    for i in range(top5_prob.size(0)):
        print(labels[top5_catid[i]], top5_prob[i].item())

main()
