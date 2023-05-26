import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from torchvision.models import resnet50
import streamlit as st

def get_prediction(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=1)
                       
    model.load_state_dict(torch.load('app_skin_care/skin_check.py', map_location=device))
    model = model.to(device)
    model.eval()

    trnsfrms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = trnsfrms(image)
    image = image.unsqueeze(0) / 255

    predictions = model(image.to(device)).sigmoid().round().cpu()
    prediction_class = predictions.item()
    if prediction_class == 0:
        return 'Benign'
    elif prediction_class == 1:
        return 'Malignant'

st.title("Skin Cancer Classification")
st.write("Skin Cancer Classification is a vital task in medical image analysis.It involves the identification and categorization of skin lesions into malignant and benign classes using machine learning techniques.") 

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = get_prediction(image)
    st.write("Prediction: ", prediction)
