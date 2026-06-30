import streamlit as st
import cv2
import av
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stress Detection",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ===== Background ===== */
.stApp{
    background: radial-gradient(circle at 20% 0%, #16213a 0%, #0b1326 55%, #070b16 100%);
    color: #E2E8F0;
}

/* ===== Hide Streamlit chrome ===== */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* ===== Header ===== */
.app-header{
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:4px;
}

.app-badge{
    width:46px;
    height:46px;
    border-radius:14px;
    background:linear-gradient(135deg,#3B82F6,#1D4ED8);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:22px;
    box-shadow:0 8px 18px rgba(29,78,216,.35);
}

.main-title{
    font-size:30px;
    font-weight:800;
    color:#F1F5F9;
    letter-spacing:-0.5px;
    margin:0;
}

.subtitle{
    color:#8B9CC0;
    font-size:15px;
    margin:0;
}

/* ===== Card ===== */
.card{
    background:linear-gradient(180deg,#101A33,#0C1428);
    padding:22px 24px;
    border-radius:16px;
    border:1px solid #1E2B4D;
}

.card h3{
    margin-top:0;
    color:#CBD5E1;
    font-size:14px;
    font-weight:600;
    letter-spacing:0.4px;
    text-transform:uppercase;
}

/* ===== Result card ===== */
.result-card{
    padding:26px;
    border-radius:18px;
    text-align:center;
    background:linear-gradient(180deg,#101A33,#0B1326);
    border:1px solid #1E2B4D;
}

.result-stress{
    border-left:4px solid #F87171;
    box-shadow:0 0 0 1px rgba(248,113,113,.08), 0 12px 24px rgba(248,113,113,.06);
}

.result-calm{
    border-left:4px solid #34D399;
    box-shadow:0 0 0 1px rgba(52,211,153,.08), 0 12px 24px rgba(52,211,153,.06);
}

.result-label{
    font-size:26px;
    font-weight:800;
    margin:6px 0 2px 0;
}

.result-stress .result-label{ color:#FCA5A5; }
.result-calm .result-label{ color:#6EE7B7; }

.result-conf{
    font-size:34px;
    font-weight:800;
    color:#F8FAFC;
    margin:8px 0 0 0;
}

.result-caption{
    color:#7C8AB0;
    font-size:13px;
    margin-top:4px;
}

/* ===== Metric ===== */
div[data-testid="stMetric"]{
    background:linear-gradient(180deg,#101A33,#0C1428);
    padding:16px 18px;
    border-radius:14px;
    border:1px solid #1E2B4D;
}

div[data-testid="stMetricLabel"]{
    color:#8B9CC0 !important;
}

div[data-testid="stMetricValue"]{
    color:#F1F5F9 !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
    background:#0A0F1F;
    border-right:1px solid #1E2B4D;
}

section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p{
    color:#A8B5D1;
}

/* ===== Tabs ===== */
button[data-baseweb="tab"]{
    color:#8B9CC0;
    font-weight:600;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#60A5FA;
}

div[data-baseweb="tab-highlight"]{
    background-color:#3B82F6;
}

/* ===== Button ===== */
.stButton>button{
    background:linear-gradient(135deg,#3B82F6,#1D4ED8);
    color:white;
    border-radius:10px;
    border:none;
    height:44px;
    font-size:15px;
    font-weight:600;
}

.stButton>button:hover{
    background:linear-gradient(135deg,#60A5FA,#2563EB);
}

/* ===== Upload ===== */
[data-testid="stFileUploader"]{
    background:#0C1428;
    padding:14px;
    border-radius:14px;
    border:1.5px dashed #2C3E66;
}

/* ===== Progress bar ===== */
div[data-testid="stProgress"] > div > div{
    background:linear-gradient(90deg,#3B82F6,#60A5FA);
}

/* ===== Caption / divider ===== */
hr{ border-color:#1E2B4D; }

.small-label{
    color:#8B9CC0;
    font-size:13px;
    font-weight:600;
    margin-bottom:2px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CNN MODEL
# ============================================================

class EmotionCNNDeepFC(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # BLOCK 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # BLOCK 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # BLOCK 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.30)
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


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    m = EmotionCNNDeepFC()
    m.load_state_dict(
        torch.load("emotion_cnn_final_fp16.pth", map_location=device)
    )

    if device.type == "cuda":
        m = m.half()

    m.to(device)
    m.eval()
    return m


model = load_model()

# ============================================================
# LABEL
# ============================================================

class_names = ["NON-STRESS", "STRESS"]

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
# PREDICT IMAGE
# ============================================================

@torch.no_grad()
def predict_image(image):
    image = np.array(image)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    if device.type == "cuda":
        image_tensor = image_tensor.half()

    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    return pred.item(), confidence.item(), probs.squeeze().cpu().numpy()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="app-header">
    <div class="app-badge">🌙</div>
    <div>
        <p class="main-title">Stress Detection</p>
        <p class="subtitle">Deteksi tingkat stres dari ekspresi wajah secara real-time</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("#### Status Model")
st.sidebar.success("Siap digunakan", icon="✅")

st.sidebar.write(f"**Device** — {device}")
st.sidebar.write(f"**Presisi** — {'FP16' if device.type == 'cuda' else 'FP32 (CPU)'}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Detail Arsitektur")
st.sidebar.write("**Model** — EmotionCNNDeepFC")
st.sidebar.write("**Input** — 48 × 48 (grayscale)")
st.sidebar.write("**Kelas** — Stress / Non-Stress")
st.sidebar.write("**Dataset** — FER2013")

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(["📁  Upload Gambar", "🎥  Webcam"])

with tab1:

    uploaded_file = st.file_uploader(
        "Unggah foto wajah (jpg / jpeg / png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, use_container_width=True)

        with col2:

            pred, conf, probs = predict_image(image)
            label = class_names[pred]

            result_class = "result-stress" if label == "STRESS" else "result-calm"
            emoji = "🚨" if label == "STRESS" else "🍃"
            display_label = "Terindikasi Stres" if label == "STRESS" else "Cenderung Tenang"

            st.markdown(f"""
            <div class="result-card {result_class}">
                <div style="font-size:28px;">{emoji}</div>
                <p class="result-label">{display_label}</p>
                <p class="result-conf">{conf*100:.1f}%</p>
                <p class="result-caption">Tingkat keyakinan model</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<p class="small-label">NON-STRESS</p>', unsafe_allow_html=True)
            st.progress(float(probs[0]))
            st.caption(f"{probs[0]*100:.2f}%")

            st.markdown('<p class="small-label">STRESS</p>', unsafe_allow_html=True)
            st.progress(float(probs[1]))
            st.caption(f"{probs[1]*100:.2f}%")

    else:
        st.markdown("""
        <div class="card">
            <h3>Belum ada gambar</h3>
            <p style="color:#8B9CC0;">Unggah foto wajah untuk mulai analisis, atau gunakan tab Webcam untuk deteksi langsung.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            try:
                face = gray[y:y + h, x:x + w]
                face_tensor = transform(face).unsqueeze(0).to(device)

                if device.type == "cuda":
                    face_tensor = face_tensor.half()

                with torch.no_grad():
                    output = model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, dim=1)

                label = class_names[pred.item()]
                conf = confidence.item() * 100

                color = (80, 80, 240) if label == "STRESS" else (130, 220, 90)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                text = f"{label} ({conf:.1f}%)"
                cv2.putText(
                    img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# WEBCAM
# ============================================================

with tab2:

    st.markdown("""
    <div class="card" style="margin-bottom:18px;">
        <h3>Deteksi Real-time</h3>
        <p style="color:#A8B5D1; margin-bottom:0;">
            Arahkan wajah ke kamera. Sistem akan mendeteksi wajah lalu mengklasifikasikan
            kondisi sebagai <b style="color:#FCA5A5;">Stress</b> atau <b style="color:#6EE7B7;">Non-Stress</b> secara langsung.
        </p>
    </div>
    """, unsafe_allow_html=True)

    webrtc_streamer(
        key="stress",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("#### Performa Model")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Akurasi Validasi", "90.00%")

with c2:
    st.metric("ROC-AUC", "0.926")

with c3:
    st.metric("Presisi", "FP16")

with c4:
    st.metric("Input", "48 × 48")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h3>Spesifikasi</h3>
    <p style="color:#A8B5D1; line-height:1.9; margin-bottom:0;">
        <b>Arsitektur</b> — EmotionCNNDeepFC &nbsp;·&nbsp;
        <b>Input</b> — 48 × 48, grayscale &nbsp;·&nbsp;
        <b>Dataset</b> — FER2013 &nbsp;·&nbsp;
        <b>Kelas</b> — Stress / Non-Stress &nbsp;·&nbsp;
        <b>Presisi</b> — FP16
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Sistem deteksi stres berbasis Convolutional Neural Network (CNN) dengan kuantisasi FP16.")