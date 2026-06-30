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
    page_title="Stressense",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp{
    background:#0B0F1A;
    color:#E2E8F0;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* ===== Brand row (sidebar) ===== */
.brand-row{
    display:flex;
    align-items:center;
    gap:8px;
    margin-bottom:18px;
}

.brand-row .ico{
    font-size:18px;
}

.brand-row .name{
    font-size:14px;
    font-weight:600;
    color:#E2E8F0;
}

.status-pill{
    display:inline-flex;
    align-items:center;
    gap:6px;
    font-size:12px;
    color:#6EE7B7;
    background:rgba(52,211,153,.10);
    border:1px solid rgba(52,211,153,.25);
    padding:4px 10px;
    border-radius:8px;
}

/* ===== Page header (main area) ===== */
.page-title{
    font-size:20px;
    font-weight:700;
    color:#F1F5F9;
    margin:0;
}

.page-sub{
    font-size:13px;
    color:#7C8AB0;
    margin:2px 0 0 0;
}

/* ===== Upload card ===== */
[data-testid="stFileUploader"]{
    background:#0F1525;
    padding:14px;
    border-radius:12px;
    border:1.5px dashed #2C3E66;
}

.upload-hint{
    font-size:12px;
    color:#5C6B8F;
    margin:6px 0 14px 2px;
}

/* ===== Bounded card shared style ===== */
.bounded-card{
    background:#0F1525;
    border:1px solid #1E2B4D;
    border-radius:12px;
    overflow:hidden;
    height:100%;
}

.preview-frame{
    aspect-ratio:4/3;
    width:100%;
    object-fit:cover;
    display:block;
}

.preview-caption{
    padding:8px 12px;
    font-size:11px;
    color:#5C6B8F;
    border-top:1px solid #1E2B4D;
}

/* ===== Result card ===== */
.result-box{
    padding:18px;
    height:100%;
    display:flex;
    flex-direction:column;
    justify-content:center;
    gap:6px;
}

.result-box.danger{ border-color:#5C2A2A; }
.result-box.calm{ border-color:#235C45; }

.result-flag{
    font-size:13px;
    display:flex;
    align-items:center;
    gap:6px;
}

.result-flag.danger{ color:#F87171; }
.result-flag.calm{ color:#34D399; }

.result-number{
    font-size:30px;
    font-weight:700;
    color:#F1F5F9;
    margin:2px 0 6px 0;
}

.result-bar-track{
    height:6px;
    border-radius:99px;
    background:#1A2238;
    overflow:hidden;
}

.result-bar-fill{
    height:100%;
    border-radius:99px;
}

.result-bar-fill.danger{ background:#F87171; }
.result-bar-fill.calm{ background:#34D399; }

/* ===== Metric grid ===== */
.metric-tile{
    background:#0F1525;
    border:1px solid #1E2B4D;
    border-radius:10px;
    padding:14px 16px;
}

.metric-label{
    font-size:11px;
    color:#5C6B8F;
    margin:0 0 4px 0;
}

.metric-value{
    font-size:18px;
    font-weight:700;
    color:#F1F5F9;
    margin:0;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
    background:#0A0E1A;
    border-right:1px solid #1A2238;
}

section[data-testid="stSidebar"] p{
    color:#8B9CC0;
}

/* ===== Tabs ===== */
button[data-baseweb="tab"]{
    color:#5C6B8F;
    font-weight:500;
    font-size:13px;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#60A5FA;
}

div[data-baseweb="tab-highlight"]{
    background-color:#3B82F6;
}

/* ===== Button ===== */
.stButton>button{
    background:#1A2238;
    color:#CBD5E1;
    border-radius:8px;
    border:1px solid #2C3E66;
    height:40px;
    font-size:13px;
    font-weight:500;
}

.stButton>button:hover{
    background:#202B45;
    border-color:#3B82F6;
    color:#F1F5F9;
}

/* ===== Footer spec strip ===== */
.spec-strip{
    font-size:11px;
    color:#475569;
    padding-top:14px;
    border-top:1px solid #1A2238;
}

hr{ border-color:#1A2238; }

</style>
""", unsafe_allow_html=True)

# ============================================================
# CNN MODEL
# ============================================================

class EmotionCNNDeepFC(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

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
# DEVICE & MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class_names = ["NON-STRESS", "STRESS"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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
# SIDEBAR — minimal nav only
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="brand-row">
        <span class="ico">🌙</span>
        <span class="name">Stressense</span>
    </div>
    <span class="status-pill">● aktif</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Navigasi**")

    st.button("📁  Unggah foto", use_container_width=True, key="nav_upload")
    st.button("🎥  Webcam", use_container_width=True, key="nav_webcam")
    st.button("📊  Riwayat", use_container_width=True, key="nav_history")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="spec-strip">
        Model EmotionCNNDeepFC · {device}<br>
        Presisi {'FP16' if device.type == 'cuda' else 'FP32'} · FER2013
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================

st.markdown("""
<p class="page-title">Deteksi stres dari wajah</p>
<p class="page-sub">Unggah foto atau gunakan webcam untuk analisis langsung.</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Unggah gambar", "Webcam"])

# ============================================================
# TAB 1 — UPLOAD, ALIGNED RESULT GRID
# ============================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Unggah foto wajah",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown(
        '<p class="upload-hint">Wajah menghadap kamera, pencahayaan cukup, format jpg/jpeg/png.</p>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        pred, conf, probs = predict_image(image)
        label = class_names[pred]
        is_stress = label == "STRESS"

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="bounded-card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(
                f'<div class="preview-caption">{uploaded_file.name} · 48 × 48 grayscale</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            tone = "danger" if is_stress else "calm"
            flag_text = "Terindikasi stres" if is_stress else "Cenderung tenang"
            flag_icon = "⚠️" if is_stress else "🍃"

            st.markdown(f"""
            <div class="bounded-card result-box {tone}">
                <div class="result-flag {tone}">{flag_icon} {flag_text}</div>
                <p class="result-number">{conf*100:.1f}%</p>
                <div class="result-bar-track">
                    <div class="result-bar-fill {tone}" style="width:{conf*100:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Rincian probabilitas"):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("NON-STRESS")
                st.progress(float(probs[0]))
                st.caption(f"{probs[0]*100:.2f}%")
            with c2:
                st.caption("STRESS")
                st.progress(float(probs[1]))
                st.caption(f"{probs[1]*100:.2f}%")

    else:
        st.markdown("""
        <div class="bounded-card" style="padding:28px; text-align:center;">
            <p style="color:#5C6B8F; font-size:13px; margin:0;">Belum ada foto diunggah</p>
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
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
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
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# TAB 2 — WEBCAM
# ============================================================

with tab2:
    st.markdown("""
    <div class="bounded-card" style="padding:16px 18px; margin-bottom:14px;">
        <p style="color:#8B9CC0; font-size:13px; margin:0;">
            Arahkan wajah ke kamera. Hasil klasifikasi muncul langsung di atas video.
        </p>
    </div>
    """, unsafe_allow_html=True)

    webrtc_streamer(
        key="stress-detection",
        video_processor_factory=VideoProcessor,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

# ============================================================
# METRIC GRID FOOTER
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

m1, m2, m3, m4 = st.columns(4)

metrics = [
    ("Akurasi", "90%"),
    ("ROC-AUC", "0.93"),
    ("Presisi", "FP16"),
    ("Input", "48 × 48"),
]

for col, (label, value) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p class="spec-strip" style="margin-top:18px;">
    EmotionCNNDeepFC · grayscale 48×48 · dataset FER2013 · kuantisasi FP16
</p>
""", unsafe_allow_html=True)
