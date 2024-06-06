import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
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

# Load an image from a folder
def load_image_from_folder(folder_path):
    try:
        # List all files in the directory
        files = os.listdir(folder_path)
        # Filter out non-image files
        image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        if not image_files:
            print("No image files found in the directory.")
            return None
        
        # Load the first image file found
        image_path = os.path.join(folder_path, image_files[0])
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        print(f"Image {image_files[0]} loaded and preprocessed successfully")
        return image
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None

# Predict the class of an image
def predict(folder_path):
    image = load_image_from_folder(folder_path)
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

# Write results to a file
def write_results_to_file(results, file_path):
    try:
        with open(file_path, "w") as f:
            for label, prob in results:
                f.write(f"{label}: {prob:.4f}\n")
        print(f"Results written to {file_path} successfully")
    except Exception as e:
        print(f"Error writing results to file: {e}")

# Main function 
def main():
    folder_path = "input_data"  # Path to the folder containing the image
    output_file_path = "output_data/results.txt"  # Path to the output file

    labels = download_and_load_labels()
    if not labels:
        print("Failed to load labels. Exiting.")
        exit()

    probabilities = predict(folder_path)
    if probabilities is None:
        print("Failed to predict. Exiting.")
        exit()

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = [(labels[top5_catid[i]], top5_prob[i].item()) for i in range(top5_prob.size(0))]
    
    write_results_to_file(results, output_file_path)

main()
