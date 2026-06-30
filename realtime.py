import streamlit as st
import cv2
import av
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================================
# MODEL CNN
# =====================================================
class EmotionCNNDeepFC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256*6*6,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =====================================================
# LOAD MODEL
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNNDeepFC()
model.load_state_dict(
    torch.load(
        "best_emotion_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

class_names = [
    "NON-STRESS",
    "STRESS"
]

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# =====================================================
# FACE DETECTOR
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================================
# PREDICT IMAGE
# =====================================================
def predict_image(image):

    image_tensor = transform(
        np.array(image)
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    return (
        pred.item(),
        confidence.item(),
        probs.cpu().numpy()[0]
    )

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="CNN Stress Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Stress Detection Dashboard")

st.write(
    """
    Sistem multimedia untuk mendeteksi kondisi stress dan non-stress
    menggunakan CNN berbasis ekspresi wajah.
    """
)

tab1, tab2 = st.tabs([
    "📁 Upload Image",
    "🎥 Realtime Webcam"
])

# =====================================================
# VIDEO PROCESSOR
# =====================================================
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            try:

                face_tensor = transform(
                    face
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = model(face_tensor)

                    probs = torch.softmax(
                        output,
                        dim=1
                    )

                    confidence, pred = torch.max(
                        probs,
                        dim=1
                    )

                label = class_names[
                    pred.item()
                ]

                conf = confidence.item() * 100

                if label == "STRESS":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                cv2.putText(
                    img,
                    f"{label} {conf:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except:
                pass

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with col2:

            pred, conf, probs = predict_image(
                image
            )

            label = class_names[pred]

            st.subheader(
                "📊 Detection Result"
            )

            if label == "STRESS":

                st.error(
                    f"🚨 {label}"
                )

            else:

                st.success(
                    f"✅ {label}"
                )

            st.metric(
                "Confidence",
                f"{conf*100:.2f}%"
            )

            st.progress(conf)

            st.subheader(
                "📈 Probability Distribution"
            )

            st.bar_chart({
                "NON-STRESS": float(probs[0]),
                "STRESS": float(probs[1])
            })
            
with tab2:

    st.subheader(
        "🎥 Realtime Webcam Detection"
    )

    st.info(
        "Arahkan wajah ke kamera. Hasil prediksi akan muncul langsung pada video."
    )

    webrtc_streamer(
        key="stress-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
    
st.divider()

st.subheader(
    "📌 Model Information"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Accuracy",
        "84.81%"
    )

with c2:
    st.metric(
        "ROC-AUC",
        "0.9264"
    )

with c3:
    st.metric(
        "Classes",
        "2"
    )

st.write("Dataset : FER2013")
st.write("Model : EmotionCNNDeepFC")
st.write("Input : 48x48 Grayscale")
st.write("Classes : Stress / Non-Stress")