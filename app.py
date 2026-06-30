import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# MODEL CNN
# =========================
class EmotionCNNDeepFC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNNDeepFC(num_classes=2)
    state_dict = torch.load("best_emotion_model.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


# =========================
# PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict_image(image, model, device):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    return predicted_class.item(), confidence.item(), probs.cpu().numpy()[0]


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="CNN Stress Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Stress Detection Dashboard")

st.write(
    """
    Sistem multimedia ini digunakan untuk mendeteksi kondisi **stress** dan **non-stress**
    berdasarkan citra ekspresi wajah menggunakan model CNN.
    """
)

st.divider()

# Load model
try:
    model, device = load_model()
    st.success(f"Model berhasil dimuat. Device: {device}")
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file `best_emotion_model.pth` ada di Colab.")
    st.exception(e)
    st.stop()


class_names = ["non_stress", "stress"]

col1, col2 = st.columns(2)

# =========================
# INPUT & RESULT LAYOUT
# =========================

left_col, right_col = st.columns([1,1])

with left_col:

    st.subheader("📷 Input Image")

    tab1, tab2 = st.tabs([
        "Upload Image",
        "Webcam"
    ])

    image = None

    with tab1:

        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["jpg","jpeg","png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    with tab2:

        camera_image = st.camera_input(
            "Take a picture"
        )

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    if image is not None:
        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

with right_col:

    st.subheader("📊 Detection Results")

    if image is not None:

        predicted_idx, confidence, probs = predict_image(
            image,
            model,
            device
        )

        predicted_label = class_names[predicted_idx]

        non_stress_prob = float(probs[0])
        stress_prob = float(probs[1])

        if predicted_label == "stress":

            st.error("🚨 STRESS DETECTED")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        else:

            st.success("✅ NON-STRESS")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.progress(float(confidence))

        st.write("### Probability Distribution")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.bar(
            ["Non-Stress","Stress"],
            [non_stress_prob, stress_prob]
        )

        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability")

        st.pyplot(fig)

        st.write("### Detailed Result")

        st.write(
            f"**Non-Stress :** {non_stress_prob:.4f}"
        )

        st.write(
            f"**Stress :** {stress_prob:.4f}"
        )

    else:

        st.info(
            "Upload image atau buka webcam terlebih dahulu."
        )


st.divider()

st.subheader("📌 Model Information")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Accuracy", "84.81%")

with colB:
    st.metric("ROC-AUC", "0.9264")

with colC:
    st.metric("Classes", "2")

st.write("Dataset : FER2013")
st.write("Model : EmotionCNNDeepFC")
st.write("Input : 48x48 Grayscale")