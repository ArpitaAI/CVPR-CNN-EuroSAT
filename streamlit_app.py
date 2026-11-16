
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EuroSAT CNN Classifier",
    page_icon="🛰️",
    layout="wide"
)

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

MODEL_PATH = Path(__file__).parent / "EuroSAT_CustomCNN_weights.pth"


# Original CNN architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class EuroSATCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, 0.05),
            ConvBlock(32, 64, 0.10),
            ConvBlock(64, 128, 0.15),
            ConvBlock(128, 256, 0.20)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Load trained model
@st.cache_resource
def load_model():
    model = EuroSATCNN(num_classes=10)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# Original evaluation preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Interface
st.title("🛰️ EuroSAT Land Cover Classification")
st.subheader("Custom Convolutional Neural Network")

st.markdown(
    "Upload a satellite image to classify its land-cover category "
    "using a custom-trained CNN."
)

st.info(
    "Model trained on the EuroSAT dataset with "
    "10 land-cover categories."
)

with st.sidebar:
    st.header("About the Model")
    st.write("Architecture: Custom CNN")
    st.write("Framework: PyTorch")
    st.write("Input size: 64 × 64 RGB")
    st.write("Number of classes: 10")
    st.write("Reported test accuracy: 94.33%")

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Classify Image", type="primary"):

        try:
            with st.spinner("Analyzing satellite image..."):

                model = load_model()

                input_tensor = transform(image).unsqueeze(0)

                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(
                        outputs, dim=1
                    )[0]

                top_values, top_indices = torch.topk(
                    probabilities, 3
                )

                predicted_index = top_indices[0].item()
                predicted_class = CLASSES[predicted_index]
                confidence = top_values[0].item() * 100

            with col2:
                st.subheader("Classification Result")

                st.success(
                    f"Predicted Class: {predicted_class}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            st.divider()
            st.subheader("Top 3 Predictions")

            for value, index in zip(top_values, top_indices):
                class_name = CLASSES[index.item()]
                score = value.item()

                st.write(
                    f"**{class_name}** — {score * 100:.2f}%"
                )

                st.progress(score)

            st.caption(
                "Confidence scores are softmax outputs and "
                "are not necessarily calibrated probabilities."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
