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
    background:#141414;
    color:#E5E5E5;
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
    color:#60A5FA;
}

.brand-row .name{
    font-size:16px;
    font-weight:700;
    color:#F5F5F5;
}

.status-label{
    font-size:12px;
    color:#9CA3AF;
    margin:0 0 6px 0;
}

.status-pill{
    display:inline-flex;
    align-items:center;
    font-size:13px;
    font-weight:600;
    color:#4ADE80;
    background:rgba(74,222,128,.12);
    padding:4px 14px;
    border-radius:6px;
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
    background:transparent;
    padding:30px 14px;
    border-radius:12px;
    border:1.5px dashed #3A3A3A;
}

[data-testid="stFileUploaderDropzone"]{
    background:transparent;
}

.upload-hint{
    font-size:12px;
    color:#6B7280;
    margin:6px 0 14px 2px;
    text-align:center;
}

/* ===== Bounded card shared style ===== */
.bounded-card{
    background:#1C1C1C;
    border:1px solid #2E2E2E;
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
    padding:10px 14px;
    font-size:12px;
    color:#9CA3AF;
    border-top:1px solid #2E2E2E;
    background:#1A1A1A;
}

.avatar-placeholder{
    aspect-ratio:4/3;
    width:100%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:40px;
    color:#4B5563;
    background:#161616;
}

/* ===== Result card ===== */
.result-box{
    padding:20px;
    height:100%;
    display:flex;
    flex-direction:column;
    justify-content:center;
    gap:8px;
}

.result-box.danger{ border-color:#5C2A2A; background:#1F1717; }
.result-box.calm{ border-color:#235C45; background:#171F1B; }

.result-flag{
    font-size:14px;
    font-weight:500;
    display:flex;
    align-items:center;
    gap:8px;
}

.result-flag.danger{ color:#F87171; }
.result-flag.calm{ color:#34D399; }

.result-number{
    font-size:32px;
    font-weight:700;
    color:#F5F5F5;
    margin:4px 0 8px 0;
}

.result-bar-track{
    height:6px;
    border-radius:99px;
    background:#2E2E2E;
    overflow:hidden;
}

.result-bar-fill{
    height:100%;
    border-radius:99px;
}

.result-bar-fill.danger{ background:#EF4444; }
.result-bar-fill.calm{ background:#22C55E; }

.detail-btn{
    margin-top:10px;
    display:inline-flex;
    align-items:center;
    gap:6px;
    font-size:13px;
    color:#E5E5E5;
    background:transparent;
    border:1px solid #3A3A3A;
    border-radius:8px;
    padding:8px 16px;
    width:fit-content;
}

/* ===== Metric grid ===== */
.metric-tile{
    background:#1C1C1C;
    border:1px solid #2E2E2E;
    border-radius:10px;
    padding:14px 16px;
}

.metric-label{
    font-size:12px;
    color:#9CA3AF;
    margin:0 0 4px 0;
}

.metric-value{
    font-size:18px;
    font-weight:700;
    color:#F5F5F5;
    margin:0;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
    background:#1A1A1A;
    border-right:1px solid #2A2A2A;
}

section[data-testid="stSidebar"] p{
    color:#9CA3AF;
}

/* ===== Tabs ===== */
button[data-baseweb="tab"]{
    color:#6B7280;
    font-weight:500;
    font-size:13px;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#60A5FA;
}

div[data-baseweb="tab-highlight"]{
    background-color:#3B82F6;
}

/* ===== Button (sidebar nav) ===== */
.stButton>button{
    background:transparent;
    color:#D1D5DB;
    border-radius:8px;
    border:1px solid #3A3A3A;
    height:42px;
    font-size:14px;
    font-weight:500;
}

.stButton>button:hover{
    border-color:#60A5FA;
    color:#F5F5F5;
}

section[data-testid="stSidebar"] button[kind="primary"]{
    background:#1D4ED8;
    border-color:#1D4ED8;
    color:white;
}

/* ===== Footer spec strip ===== */
.spec-strip{
    font-size:12px;
    color:#6B7280;
    padding-top:14px;
    border-top:1px solid #2A2A2A;
}

hr{ border-color:#2A2A2A; }

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
        <span class="ico">◎</span>
        <span class="name">Stressense</span>
    </div>
    <p class="status-label">Status</p>
    <span class="status-pill">Aktif</span>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.button("🖼️  Unggah foto", use_container_width=True, key="nav_upload", type="primary")
    st.button("📷  Webcam", use_container_width=True, key="nav_webcam")
    st.button("📶  Riwayat", use_container_width=True, key="nav_history")

    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="spec-strip">
        Detail model di footer, bukan di sidebar utama
    </div>
    """, unsafe_allow_html=True)

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
            flag_icon = "⚠" if is_stress else "🍃"

            st.markdown(f"""
            <div class="bounded-card result-box {tone}">
                <div class="result-flag {tone}">{flag_icon} {flag_text}</div>
                <p class="result-number">{conf*100:.1f}%</p>
                <div class="result-bar-track">
                    <div class="result-bar-fill {tone}" style="width:{conf*100:.1f}%;"></div>
                </div>
                <div class="detail-btn">Lihat detail ↗</div>
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
        key="stress",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ============================================================
# METRIC GRID FOOTER
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

m1, m2, m3 = st.columns(3)

metrics = [
    ("Akurasi", "90%"),
    ("Presisi", "FP16"),
    ("Latency", "~40ms"),
]

for col, (label, value) in zip([m1, m2, m3], metrics):
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
