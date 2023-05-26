import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision import transforms

def main():
    st.title("Cat vs. Dog Classifier")
    st.text("Upload an image and the classifier will determine if it's a cat or a dog.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)


        # Preprocess the image
        img_tensor = preprocess_image(image)

        # Make predictions
        output = model(img_tensor.unsqueeze(0)).sigmoid().round()
        
        class_label = get_class_label(output.item())

        # Display the predicted class
        st.success(f"The uploaded image is a {class_label}.")

def get_class_label(class_label):
    if class_label == 0:
        return "cat"
    elif class_label == 1:
        return "dog"
    else:
        return "unknown"

def preprocess_image(image):
    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transforms_pipeline(image)

if __name__ == "__main__":
    # Load the pre-trained model
    PATH = '/Users/vasevooo/projects/nn_project/resnet50_dogs_cats.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet50()
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    main()
