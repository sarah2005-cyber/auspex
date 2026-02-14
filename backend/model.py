import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import subprocess
import tempfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class FixedHPF(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[[-1.0, 2.0, -1.0]]]])
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x_padded = F.pad(x, (1, 1, 0, 0), mode='reflect')
        return F.conv2d(x_padded, self.weight)

class SFFN_1D_Stream(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 9), padding=(0, 4)),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.spatial_pool = nn.AdaptiveMaxPool2d((8, 8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spatial_pool(x)
        return x.view(x.size(0), -1)

class SFFNLite_Universal(nn.Module):
    def __init__(self, stats_1024=(1.0754, 1.3645), stats_512=(0.4015, 0.5033)):
        super().__init__()
        self.register_buffer("m1", torch.tensor(stats_1024[0]))
        self.register_buffer("std1", torch.tensor(stats_1024[1]))
        self.register_buffer("m2", torch.tensor(stats_512[0]))
        self.register_buffer("std2", torch.tensor(stats_512[1]))
        self.hpf = FixedHPF()
        self.s1024 = SFFN_1D_Stream()
        self.s512 = SFFN_1D_Stream()
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(4096, 2)

    def forward(self, x1, x2):
        x1 = (x1 - self.m1) / (self.std1 + 1e-6)
        x2 = (x2 - self.m2) / (self.std2 + 1e-6)
        f1 = self.s1024(self.hpf(x1))
        f2 = self.s512(self.hpf(x2))
        
        alpha_1024 = torch.sigmoid(self.fusion_weight)
        alpha_512 = 1 - alpha_1024
        
        fused = alpha_1024 * f1 + alpha_512 * f2
        logits = self.classifier(fused)
        
        # Returns intrinsic XAI data (alphas)
        return logits, {'alpha_1024': alpha_1024.item(), 'alpha_512': alpha_512.item()}

# ==========================================
# 2. XAI EXPLAINER CLASS 
# ==========================================
class ForensicExplainer:
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    def saliency_map(self, x1, x2, target=1):
        """Generates attribution maps using gradient-based saliency (fast)."""
        x1 = x1.clone().requires_grad_(True)
        x2 = x2.clone().requires_grad_(True)
        
        # Forward pass
        logits, _ = self.model(x1, x2)
        score = logits[:, target].sum()
        
        # Backward pass
        score.backward()
        
        # Saliency is the absolute gradient
        sal1 = x1.grad.abs().detach()
        sal2 = x2.grad.abs().detach()
        
        return sal1, sal2

# ==========================================
# 3. ROBUST PREPROCESSING
# ==========================================
def preprocess_for_streamlit(file_path):
    target_sr = 44100
    temp_dir = tempfile.mkdtemp()
    decoded_path = os.path.join(temp_dir, "decoded.wav")
    
    # Decoding logic matching notebook
    cmd = ["ffmpeg", "-y"]
    if file_path.endswith(".pcm"):
        cmd.extend(["-f", "s16le", "-ar", "8000", "-ac", "1"])
    elif file_path.endswith(".g729a"):
        cmd.extend(["-f", "g729"])
    cmd.extend(["-i", file_path, "-ar", str(target_sr), "-ac", "1", decoded_path])
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    y, sr = librosa.load(decoded_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Forensic Standardization
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0)
    y = np.clip(y, -1.0, 1.0)

    # Spectrogram Extraction
    def get_spec(audio, n_fft, hop, target_frames):
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop, window="hamming"))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        if spec.shape[1] < target_frames:
            spec = np.pad(spec, ((0, 0), (0, target_frames - spec.shape[1])))
        else:
            spec = spec[:, :target_frames]
        return spec

    spec1024 = get_spec(y, 1024, 512, 87)
    spec512 = get_spec(y, 512, 256, 173)
    
    # Cleanup
    if os.path.exists(decoded_path): os.remove(decoded_path)
    os.rmdir(temp_dir)
    return spec1024, spec512

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Universal Audio Steganalysis", layout="wide")
st.title("🛡️ Forensic Audio Steganalysis")

# Load Model
@st.cache_resource
def load_trained_model():
    ckpt_path = "Auspex_Universal_v1_seed2026_best_epoch.pt"
    model = SFFNLite_Universal()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # Filter unexpected keys
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    return None

model = load_trained_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "g729a", "pcm"])

if uploaded_file and model:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tfile:
        tfile.write(uploaded_file.read())
        tpath = tfile.name

    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button(" Detect Steganography", use_container_width=True)
    with col2:
        explain_btn = st.button(" Explain Decision (XAI)", use_container_width=True)

    if predict_btn or explain_btn:
        with st.spinner("Processing Audio..."):
            s1024, s512 = preprocess_for_streamlit(tpath)
            t1024 = torch.from_numpy(s1024).unsqueeze(0).unsqueeze(0).float().to(device)
            t512 = torch.from_numpy(s512).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Inference
            with torch.no_grad():
                logits, attn = model(t1024, t512)
                probs = F.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()

        # --- RESULT DISPLAY ---
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.subheader("Verdict")
            if pred == 1:
                st.error(" STEGO DETECTED")
            else:
                st.success(" CLEAN AUDIO")
            
            st.metric("Confidence", f"{probs[pred]*100:.2f}%")
            
            # Intrinsic XAI Display
            st.markdown("**Intrinsic Model Reliance:**")
            st.progress(attn['alpha_1024'], text=f"1024-Stream (Texture): {attn['alpha_1024']:.2f}")
            st.progress(attn['alpha_512'], text=f"512-Stream (Temporal): {attn['alpha_512']:.2f}")

        with res_col2:
            st.subheader("Input Spectrograms")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.imshow(s1024, aspect='auto', origin='lower', cmap='magma'); ax1.set_title("1024-FFT")
            ax2.imshow(s512, aspect='auto', origin='lower', cmap='magma'); ax2.set_title("512-FFT")
            st.pyplot(fig)

        # --- POST-HOC XAI DISPLAY ---
        if explain_btn:
            st.divider()
            st.subheader(" Post-Hoc Forensic Analysis")
            with st.spinner("Generating Saliency Maps..."):
                explainer = ForensicExplainer(model, device)
                sal1, sal2 = explainer.saliency_map(t1024, t512, target=pred)
                
                # Visualize Attributions
                
                fig_xai, (xax1, xax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # 1024 Stream Attribution
                im1 = xax1.imshow(sal1.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='jet')
                xax1.set_title("Feature Importance (1024 Stream)")
                plt.colorbar(im1, ax=xax1)
                
                # 512 Stream Attribution
                im2 = xax2.imshow(sal2.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='jet')
                xax2.set_title("Feature Importance (512 Stream)")
                plt.colorbar(im2, ax=xax2)
                
                st.pyplot(fig_xai)
                st.info("The 'Jet' heatmaps above show which time-frequency bins contributed most to the decision. Red/Orange areas are highly suspicious.")

    # Cleanup
    if os.path.exists(tpath): os.remove(tpath)