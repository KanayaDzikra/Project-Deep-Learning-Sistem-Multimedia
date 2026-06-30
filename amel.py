import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================
# CSS
# =====================================
st.markdown("""
<style>
.stApp{
    background-color:#0f172a;
}
.title{
    color:white;
    font-size:42px;
    font-weight:800;
    margin-bottom:0px;
}
.subtitle{
    color:#94a3b8;
    font-size:16px;
    margin-bottom:25px;
}
.block{
    background:#1e293b;
    padding:25px;
    border-radius:18px;
    border:1px solid #334155;
}
.result-card{
    padding:25px;
    border-radius:18px;
    text-align:center;
    margin-bottom:15px;
}
.result-positive{
    background:linear-gradient(135deg, #064e3b, #065f46);
    border:1px solid #10b981;
}
.result-negative{
    background:linear-gradient(135deg, #450a0a, #7f1d1d);
    border:1px solid #ef4444;
}
.result-label{
    font-size:32px;
    font-weight:800;
    color:white;
    margin:0px;
}
.result-conf{
    color:#cbd5e1;
    font-size:15px;
}
.class-row{
    color:#e2e8f0;
    font-size:14px;
    margin-top:15px;
    margin-bottom:4px;
}
.placeholder{
    color:#64748b;
    text-align:center;
    padding:60px 20px;
    border:2px dashed #334155;
    border-radius:18px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# MODEL — arsitektur sudah disesuaikan dengan bobot checkpoint
# =====================================
class EmotionCNNDeepFC(nn.Module):
    def __init__(self):
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
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        st.error(f"❌ File model tidak ditemukan: `{path}`")
        st.stop()

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error(f"❌ Gagal load file model: {e}")
        st.stop()

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.eval()
        return model

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = EmotionCNNDeepFC()

    try:
        model.load_state_dict(cleaned_state_dict)
    except Exception as e:
        st.error(f"❌ State dict tidak cocok dengan arsitektur model: {e}")
        st.stop()

    model.eval()
    return model

# =====================================
# TRANSFORM
# =====================================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor()
])

classes = ["Negative", "Positive"]
class_emoji = {"Negative": "😔", "Positive": "😊"}

# =====================================
# HEADER
# =====================================
st.markdown('<p class="title">Emotion Recognition System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN Deep Fully Connected — Facial Stress Detection</p>', unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    model_option = st.selectbox("Pilih Model", ["FP32", "FP16"])

    if model_option == "FP32":
        model_path = os.path.join(BASE_DIR, "best_emotion_model(2).pth")
    else:
        model_path = os.path.join(BASE_DIR, "emotion_cnn_final_fp16.pth")

    st.divider()
    with st.expander("ℹ️ Tentang Model"):
        st.write(
            "Model CNN custom (EmotionCNNDeepFC) dilatih pada dataset FER2013 "
            "untuk klasifikasi biner kondisi emosi wajah: **Negative** vs **Positive**."
        )

model = load_model(model_path)

# =====================================
# MAIN
# =====================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 📤 Upload Gambar")
    uploaded = st.file_uploader("Pilih foto wajah", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True, caption="Gambar yang diupload")

with col2:
    st.markdown("#### 📊 Hasil Prediksi")

    if uploaded:
        try:
            with st.spinner("Menganalisis ekspresi wajah..."):
                img = transform(image)
                img = img.unsqueeze(0)

                with torch.no_grad():
                    output = model(img)
                    prob = torch.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)

            label = classes[pred.item()]
            confidence = conf.item() * 100
            card_class = "result-positive" if label == "Positive" else "result-negative"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <p class="result-label">{class_emoji[label]} {label}</p>
                <p class="result-conf">Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probabilitas per kelas:**")
            for i, cls in enumerate(classes):
                p = prob[0][i].item()
                st.markdown(f'<p class="class-row">{class_emoji[cls]} {cls} — {p*100:.2f}%</p>', unsafe_allow_html=True)
                st.progress(p)

        except Exception as e:
            st.error(f"❌ Terjadi error saat inferensi: {e}")

    else:
        st.markdown(
            '<div class="placeholder">📷<br>Upload gambar dulu di sebelah kiri<br>buat lihat hasil prediksinya</div>',
            unsafe_allow_html=True
        )