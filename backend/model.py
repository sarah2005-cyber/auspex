import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from captum.attr import IntegratedGradients
from datetime import datetime

# Try to import FPDF for PDF generation
try:
    from fpdf import FPDF
except ImportError:
    st.error("Missing dependency: fpdf. Please run 'pip install fpdf' to enable reporting.")

# ==========================================
# 0. REFINED CSS (Balanced Text & Spaced Tabs)
# ==========================================
st.set_page_config(page_title="Auspex Forensic AI", layout="wide")

st.markdown(
    """
    <style>
    /* Global font size - Reduced from 1.3 to 1.1 */
    html, body, [class*="st-"] { font-size: 1.1rem; }
    
    /* Titles - Scaled down for professionalism */
    h1 { font-size: 3rem !important; font-weight: 800 !important; color: #1E3A8A; margin-bottom: 0.2rem; }
    h2 { font-size: 2.2rem !important; border-bottom: 2px solid #1E3A8A; padding-bottom: 8px; }
    h3 { font-size: 1.8rem !important; color: #1E3A8A; }

    /* Space out Tabs heavily - Increased margin */
    button[data-baseweb="tab"] {
        margin-right: 80px !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }

    /* Professional Buttons */
    div.stButton > button:first-child {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        height: 3em !important;
        border-radius: 10px;
        background-color: #1E3A8A;
        color: white;
    }

    /* Alerts */
    .stAlert p { font-size: 1.2rem !important; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# 1. CORE ARCHITECTURE (Dual-Pathway v32)
# ============================================================================

class DualPathwayForensicLayer(nn.Module):
    def __init__(self):
        super().__init__()
        hp1 = torch.tensor([[[[-1., 1.]]]], dtype=torch.float32)
        hp2 = torch.tensor([[[[-1., 2., -1.]]]], dtype=torch.float32)
        hp3 = torch.tensor([[[[-1., 3., -3., 1.]]]], dtype=torch.float32)
        hp_spatial = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]], dtype=torch.float32)
        self.register_buffer('kernel1', hp1); self.register_buffer('kernel2', hp2)
        self.register_buffer('kernel3', hp3); self.register_buffer('kernel_spatial', hp_spatial)
        self.learnable_filters = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        nn.init.xavier_normal_(self.learnable_filters.weight)

    def forward(self, x):
        raw_bits = x[:, 0:1, :, :]
        res1 = F.pad(F.conv2d(raw_bits, self.kernel1, padding=0), (0, 1, 0, 0))
        res2 = F.pad(F.conv2d(raw_bits, self.kernel2, padding=0), (1, 1, 0, 0))
        res3 = F.pad(F.conv2d(raw_bits, self.kernel3, padding=0), (1, 2, 0, 0))
        res_sp = F.conv2d(raw_bits, self.kernel_spatial, padding=1)
        return torch.cat([res1, res2, res3, res_sp, self.learnable_filters(raw_bits)], dim=1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels); self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        return F.relu(self.bn2(self.pointwise(self.dropout(F.relu(self.bn1(self.depthwise(x)))))))

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels); self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        res = x
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + res)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1).expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn, attn

class StegoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hp_layer = DualPathwayForensicLayer()
        self.init_conv = nn.Sequential(nn.Conv2d(9, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.MaxPool2d(2, 2))
        self.dw_conv = DepthwiseSeparableConv(32, 64); self.pool2 = nn.MaxPool2d(2, 2)
        self.attention = SpatialAttention()
        self.res_block1 = ResidualBlock(64); self.se1 = SEBlock(64)
        self.res_block2 = ResidualBlock(64); self.se2 = SEBlock(64)
        self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, 1))
    def forward(self, x):
        hp = self.hp_layer(x); x = torch.cat([x, hp], dim=1)
        x = self.init_conv(x); x = self.pool2(self.dw_conv(x))
        x, attn = self.attention(x); x = self.se1(self.res_block1(x)); x = self.se2(self.res_block2(x))
        pool = torch.cat([torch.mean(x, dim=[2, 3]), torch.std(x, dim=[2, 3])], dim=1)
        return self.fc(pool), attn

# ==========================================
# 2. TIGHTENED REPORTING ENGINE
# ==========================================

def create_pro_report(verdict, prob, sample_name, bitstream_img_path, heatmap_img_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Section - Tightened height
    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", 'B', 20)
    pdf.cell(0, 15, "AUSPEX: FORENSIC INVESTIGATION REPORT", ln=True, align='C')
    pdf.set_font("Helvetica", 'I', 9)
    pdf.cell(0, 5, f"Investigator: Sarah Rahim | {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')

    # 1. VERDICT - Reduced padding
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, " 1. FINAL FORENSIC VERDICT", ln=True, fill=True)
    
    if verdict == "STEGO":
        pdf.set_text_color(180, 0, 0)
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 12, f"STATUS: COMPROMISED (STEGO DETECTED)", ln=True, align='C')
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 12, f"STATUS: CLEAN (COVER SAMPLE)", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 8, f"Confidence: {prob*100:.2f}% | Sample: {sample_name}", ln=True, align='C')

    # 2. EVIDENCE - Tightened Exhibit flow
    pdf.ln(4)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, " 2. MULTI-VIEW FORENSIC EVIDENCE", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "Exhibit A: Bitstream Decomposition (Flux & Stability)", ln=True)
    pdf.image(bitstream_img_path, x=10, w=180) # Scaled down slightly for fit
    
    pdf.ln(4)
    pdf.cell(0, 6, "Exhibit B: Manipulation Hotspots & Neural Reliance", ln=True)
    pdf.image(heatmap_img_path, x=10, w=180)

    # 3. CONCLUSION - No extra page, just flows down
    pdf.ln(4)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, " 3. INVESTIGATIVE CONCLUSION", ln=True, fill=True)
    pdf.set_font("Helvetica", '', 10)
    
    cert = "high" if prob > 0.8 or prob < 0.2 else "moderate"
    if verdict == "STEGO":
        conclusion = f"Classification: STEGO ({cert} certainty). Exhibit B confirms localized entropy anomalies consistent with embedding flux."
    else:
        conclusion = f"Classification: CLEAN ({cert} certainty). Exhibit A demonstrates bitstream stability within standard encoder tolerances."
    
    pdf.multi_cell(0, 6, conclusion)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. UTILS & PROCESSING
# ==========================================

def preprocess_input(file_path):
    if file_path.endswith('.npy'):
        matrix = np.load(file_path).astype(np.float32)
        if matrix.ndim == 3 and matrix.shape[0] == 3: return matrix
        raw_matrix = matrix if matrix.ndim == 2 else matrix.squeeze()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
        if not data: return None
        data = data[:1000].ljust(1000, b'\x00')
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        raw_matrix = bits.reshape(100, 80).astype(np.float32)

    diff = np.zeros_like(raw_matrix)
    diff[1:] = np.abs(raw_matrix[1:] - raw_matrix[:-1])
    stability = np.abs(raw_matrix - np.mean(raw_matrix, axis=0, keepdims=True))
    return np.stack([raw_matrix, diff, stability], axis=0)

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.title("🛡️ AUSPEX: Forensic Dashboard")

@st.cache_resource
def load_model(path):
    model = StegoCNN()
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)
            return model.eval()
        except Exception as e:
            st.error(f"Load Error: {e}")
    return None

model_path = st.sidebar.text_input("Model Checkpoint", "Auspex_Forensic_Final_Original_seed42_best.pt")
uploaded_file = st.file_uploader("Upload Bitstream", type=["g729a", "npy"])

model = load_model(model_path)

if uploaded_file and model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer()); tmp_path = tmp.name

    try:
        tensor_np = preprocess_input(tmp_path)
        input_tensor = torch.from_numpy(tensor_np).unsqueeze(0)

        tab1, tab2 = st.tabs(["🚀 QUICK SCAN", "🧬 DEEP ANALYSIS"])

        with tab1:
            st.write("### Sub-perceptual Integrity Check")
            if st.button("RUN SCAN"):
                logits, _ = model(input_tensor); prob = torch.sigmoid(logits).item()
                verdict = "STEGO" if prob > 0.44 else "COVER"
                if verdict == "STEGO": st.error(f"🚨 {verdict} DETECTED | {prob*100:.2f}% Confidence")
                else: st.success(f"✅ {verdict} | {(1-prob)*100:.2f}% Confidence")
                st.progress(prob)

        with tab2:
            st.write("### Evidence Extraction")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                explain_btn = st.button("GENERATE EVIDENCE")
            
            if explain_btn:
                with st.spinner("Analyzing..."):
                    logits, attn = model(input_tensor); prob = torch.sigmoid(logits).item()
                    verdict = "STEGO" if prob > 0.44 else "COVER"

                    with tempfile.TemporaryDirectory() as img_dir:
                        # Exhibits
                        fig_bits, ax_bits = plt.subplots(1, 3, figsize=(14, 4))
                        for i, title in enumerate(["Raw", "Flux", "Stability"]):
                            ax_bits[i].imshow(tensor_np[i], cmap='binary', aspect='auto')
                            ax_bits[i].set_title(title, fontsize=12); ax_bits[i].axis('off')
                        bit_path = os.path.join(img_dir, "b.png")
                        fig_bits.savefig(bit_path, dpi=120); plt.close(fig_bits)

                        fig_xai, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        attn_map = F.interpolate(attn, size=(100, 80), mode='bilinear').squeeze().detach().numpy()
                        ax1.imshow(attn_map, cmap='hot', aspect='auto')
                        ax1.set_title("Hotspots", fontsize=14); ax1.axis('off')
                        
                        ig = IntegratedGradients(lambda x: model(x)[0])
                        attr = ig.attribute(input_tensor, target=0).squeeze().cpu().detach().numpy()
                        imp = np.abs(attr).sum(axis=(1, 2))
                        ax2.bar(["Raw", "Flux", "Stab"], (imp/imp.sum())*100, color=['#e63946', '#457b9d', '#2a9d8f'])
                        ax2.set_title("Reliance %", fontsize=14)
                        xai_path = os.path.join(img_dir, "x.png")
                        fig_xai.savefig(xai_path, dpi=120)

                        with col_b:
                            pdf_bytes = create_pro_report(verdict, prob, uploaded_file.name, bit_path, xai_path)
                            st.download_button(label="📥 DOWNLOAD PDF REPORT", data=pdf_bytes, file_name=f"Report_{uploaded_file.name.split('.')[0]}.pdf", mime="application/pdf")
                        
                        st.divider()
                        st.write(f"#### Final Verdict: **{verdict}**")
                        st.pyplot(fig_xai)
                        st.pyplot(fig_bits)

    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)