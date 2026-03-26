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
from fpdf import FPDF
from datetime import datetime
from PIL import Image

# ==========================================
# 0. UI CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Auspex Forensic AI", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="st-"] { font-size: 1.1rem; }
    h1 { font-size: 3rem !important; font-weight: 800 !important; color: #1E3A8A; margin-bottom: 0.2rem; }
    h2 { font-size: 2.2rem !important; border-bottom: 2px solid #1E3A8A; padding-bottom: 8px; }
    button[data-baseweb="tab"] { margin-right: 80px !important; padding: 10px 20px !important; }
    button[data-baseweb="tab"] p { font-size: 1.4rem !important; font-weight: 600 !important; }
    div.stButton > button:first-child { 
        font-size: 1.2rem !important; font-weight: bold !important; 
        height: 3em !important; border-radius: 10px; background-color: #1E3A8A; color: white; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# 1. CORE ARCHITECTURE (StegoCNN - 9 Channel Core)
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualPathwayForensicLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel1', torch.tensor([[[[-1., 1.]]]], dtype=torch.float32))
        self.register_buffer('kernel2', torch.tensor([[[[-1., 2., -1.]]]], dtype=torch.float32))
        self.register_buffer('kernel3', torch.tensor([[[[-1., 3., -3., 1.]]]], dtype=torch.float32))
        self.register_buffer('kernel_spatial', torch.tensor([[[[-1., -1., -1.], [ -1.,  8., -1.], [-1., -1., -1.]]]], dtype=torch.float32))
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
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x))); out = self.dropout(out)
        return F.relu(self.bn2(self.conv2(out)) + res)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
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
        hp = self.hp_layer(x); x = torch.cat([x, hp], dim=1); x = self.init_conv(x)
        x = self.pool2(self.dw_conv(x)); x, attn = self.attention(x)
        x = self.se1(self.res_block1(x)); x = self.se2(self.res_block2(x))
        return self.fc(torch.cat([torch.mean(x, dim=[2, 3]), torch.std(x, dim=[2, 3])], dim=1)), attn
    
# ============================================================================
# 2. CALIBRATED FORENSIC REPORTING ENGINE
# ============================================================================

class AuspexReporter:
    def __init__(self, model, device, threshold=0.5229):
        self.model = model.to(device).eval(); self.device = device
        self.threshold = threshold; self.ig = IntegratedGradients(lambda x: self.model(x)[0])
        self.channel_names = ['Raw Bits', 'Temp. Diff', 'Bit Stability']
        self.img_w = 130; self.small_img_w = 90
        self.center_x = (210 - self.img_w) / 2
        self.center_x_small = (210 - self.small_img_w) / 2

    def generate_report_bytes(self, tensor, sample_name, img_dir):
        input_t = tensor.clone().requires_grad_(True)
        with torch.no_grad():
            logits, _ = self.model(input_t)
            prob = torch.sigmoid(logits).item()
        
        attr = self.ig.attribute(input_t, target=0, n_steps=50).squeeze(0).cpu().detach().numpy()
        img_paths = self._create_plots(tensor.squeeze(0), attr, img_dir)

        if prob > (self.threshold + 0.1): risk, status, b_color = "HIGH RISK", "COMPROMISED", (231, 76, 60)
        elif prob > self.threshold: risk, status, b_color = "SUSPICIOUS", "COMPROMISED", (231, 76, 60)
        elif prob > (self.threshold - 0.1): risk, status, b_color = "UNCERTAIN", "INCONCLUSIVE", (241, 196, 15)
        else: risk, status, b_color = "LIKELY CLEAN", "CLEAN", (46, 204, 113)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(30, 35, 45); pdf.rect(0, 0, 210, 30, 'F')
        pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", 'B', 18)
        pdf.cell(190, 10, "AUSPEX: BLIND FORENSIC AUDIT", ln=True, align='C')
        pdf.set_font("Helvetica", 'I', 8); pdf.cell(190, 5, f"Automated Field Scan | Neural Core v5.8 | Threshold: {self.threshold}", ln=True, align='C')
        pdf.ln(10)

        # [1] Metadata
        pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", 'B', 10); pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 7, " [1] ANALYSIS CONTEXT", ln=True, fill=True); pdf.set_font("Courier", '', 9); pdf.ln(1)
        pdf.cell(190, 5, f" > SAMPLE ID: {sample_name}", ln=True)
        pdf.set_font("Courier", 'B', 9); pdf.cell(190, 5, f" > INITIAL FIELD VERDICT: {status}", ln=True); pdf.ln(2)

        # [2] Calibrated Gauge
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [2] CALIBRATED RISK ASSESSMENT", ln=True, fill=True); pdf.ln(3)
        pdf.set_x(45); pdf.set_font("Helvetica", 'B', 7)
        pdf.cell(40, 3, "LIKELY CLEAN", align='L'); pdf.cell(40, 3, "UNCERTAIN", align='C'); pdf.cell(40, 3, "HIGH RISK", align='R', ln=True) 
        pdf.set_draw_color(200, 200, 200); pdf.rect(45, pdf.get_y(), 120, 4); pdf.set_fill_color(*b_color); pdf.rect(45, pdf.get_y(), 120 * prob, 4, 'F') 
        pdf.set_draw_color(0, 0, 0); pdf.set_line_width(0.5); pdf.line(45 + (120 * self.threshold), pdf.get_y()-1, 45 + (120 * self.threshold), pdf.get_y()+5)
        pdf.set_y(pdf.get_y() + 5); pdf.set_font("Helvetica", 'B', 9)
        pdf.cell(190, 5, f"Neural Score: {prob:.4f} | Forensic Status: {status}", align='C', ln=True); pdf.ln(3)

        # [3] Scans
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [3] PRIMARY ARTIFACT VISUALIZATION", ln=True, fill=True); pdf.ln(2)
        pdf.image(img_paths[0], x=self.center_x, y=None, w=self.img_w); pdf.ln(4)
        pdf.image(img_paths[1], x=self.center_x, y=None, w=self.img_w)

        # Page 2
        pdf.add_page(); pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [4] EXPLAINABLE DECISION EVIDENCE", ln=True, fill=True); pdf.ln(3)
        pdf.image(img_paths[2], x=self.center_x, y=None, w=self.img_w); pdf.ln(4)
        pdf.image(img_paths[3], x=self.center_x, y=None, w=self.img_w); pdf.ln(8)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [5] CHANNEL ATTRIBUTION WEIGHT", ln=True, fill=True); pdf.ln(3)
        pdf.image(img_paths[4], x=self.center_x_small, y=None, w=self.small_img_w)

        # Page 3
        pdf.add_page(); pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [6] INVESTIGATIVE GUIDANCE", ln=True, fill=True); pdf.set_font("Helvetica", '', 9); pdf.ln(2)
        guides = [("1. Signal Pattern", "Observe vertical bit-distribution symmetry."),
                  ("2. Forensic Residual", "Attention: Bright clusters indicate algorithmic anomalies."),
                  ("3. Neural Focus", "Attention: Clustered Cyan hotspots reveal verdict-driving bits."),
                  ("4. Timeline", "Attention: Sharp spikes reveal payload temporal location."),
                  ("5. Channel Intensity", "Identify the dominant channel.")]
        for title, guidance in guides:
            pdf.set_font("Helvetica", 'B', 9); pdf.cell(190, 5, f" - {title}", ln=True)
            pdf.set_font("Helvetica", '', 8.5); pdf.cell(190, 5, f"   Tip: {guidance}", ln=True); pdf.ln(1)

        pdf.ln(3); pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [7] PIPELINE LIMITATIONS", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Helvetica", 'I', 8)
        lims = ["1. Sensitivity: Detection accuracy degrades below 0.1 bpc.", 
                "2. Attribution: Map shows neural influence, not direct payload extraction."]
        for l in lims: pdf.cell(190, 4, f" - {l}", ln=True)

        pdf.ln(4); pdf.set_font("Helvetica", 'B', 11); pdf.cell(190, 8, " [8] INVESTIGATIVE FINDINGS", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Helvetica", '', 10)
        msg = "Findings: Indicators suggest pattern manipulation consistent with steganography." if prob > self.threshold else "Findings: Signal characteristics remain within natural variance."
        pdf.multi_cell(190, 5, f"{msg} Forensic secondary audit recommended." if prob > self.threshold else f"{msg} No further action required.")


        pdf.ln(5); pdf.set_font("Helvetica", 'B', 15); pdf.set_text_color(*b_color)
        pdf.cell(190, 15, f"!!! FINAL FIELD VERDICT: {status} !!!", border=1, ln=True, align='C')
        
        # Return as Byte String
        return pdf.output(dest='S').encode('latin-1')

    def _create_plots(self, tensor, attr, img_dir):
        paths = [os.path.join(img_dir, f"v{i}.png") for i in range(1, 6)]
        def save_f(path, title, data, cmap=None, is_bar=False, sz=(10, 6)):
            plt.figure(figsize=sz)
            if is_bar:
                plt.bar(self.channel_names, data, color=['#5D6D7E', '#EB984E', '#17A589'], edgecolor='black')
                for i, v in enumerate(data): plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', weight='bold')
                plt.ylabel("Influence (%)")
            elif cmap:
                if "XAI" in title: # Add background to hotspots
                    plt.imshow(tensor[0].cpu().numpy(), cmap='gray', alpha=0.3, aspect='auto')
                    mask = np.abs(attr[0]) > np.percentile(np.abs(attr[0]), 98)
                    data = np.ma.masked_where(~mask, attr[0])
                im = plt.imshow(data, cmap=cmap, aspect='auto'); plt.colorbar(im, fraction=0.046, pad=0.04); plt.axis('off')
            else:
                plt.fill_between(range(len(data)), data, color='teal', alpha=0.3); plt.plot(data, color='teal')
                plt.ylabel("Intensity")
            plt.title(title, fontweight='bold', fontsize=14); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()

        save_f(paths[0], "1. Bitstream Scan", tensor[0].cpu().numpy(), cmap='gray')
        hp_k = torch.tensor([[[[-1., -1., -1.], [ -1.,  8., -1.], [-1., -1., -1.]]]]).to(self.device)
        res = F.conv2d(tensor.unsqueeze(0)[:, 0:1, :, :], hp_k, padding=1).squeeze().cpu().numpy()
        save_f(paths[1], "2. Forensic Residual Scan", res, cmap='inferno')
        save_f(paths[2], "3. Neural Decision Focus (XAI)", attr[0], cmap='cool')
        save_f(paths[3], "4. Temporal Attribution Timeline", np.abs(attr).sum(axis=(0, 2)))
        imp = np.abs(attr).sum(axis=(1, 2)); save_f(paths[4], "5. Attribution Intensity by Channel", (imp/imp.sum())*100, is_bar=True, sz=(7, 4))
        return paths

# ==========================================
# 3. UTILS & PROCESSING
# ==========================================

def preprocess_input(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()[:1000]
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).reshape(-1, 80).astype(np.float32)
    bits = bits[:100, :] 
    diff = np.zeros_like(bits); diff[1:] = np.abs(bits[1:] - bits[:-1])
    stability = np.abs(bits - np.mean(bits, axis=0, keepdims=True))
    return np.stack([bits, diff, stability], axis=0)

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.title("🛡️ AUSPEX: Forensic Dashboard")

@st.cache_resource
def load_model(path):
    model = StegoCNN()
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        return model.eval().to(DEVICE)
    return None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "Auspex_Forensic_Final_Original_seed42_best.pt")
model_path = st.sidebar.text_input("Model Checkpoint", DEFAULT_MODEL_PATH)
uploaded_file = st.file_uploader("Upload G.729a Bitstream", type=["g729a", "npy"])

model = load_model(model_path)

if 'scanned' not in st.session_state: st.session_state.scanned = False
if 'current_file' not in st.session_state: st.session_state.current_file = ""

if uploaded_file and model:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.scanned = False
        st.session_state.ev_generated = False
        st.session_state.current_file = uploaded_file.name

    tab1, tab2 = st.tabs(["🚀 QUICK SCAN", "🧬 DEEP ANALYSIS"])

    with tab1:
        st.write("### Sub-perceptual Integrity Check")
        if st.button("RUN INITIAL SCAN"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer()); tmp_path = tmp.name
                
                tensor_np = preprocess_input(tmp_path)
                st.session_state.input_tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits, _ = model(st.session_state.input_tensor)
                    prob = torch.sigmoid(logits).item()
                
                st.session_state.prob = prob
                if prob > 0.5229:
                    st.session_state.verdict = "STEGO"
                    st.session_state.display_conf = prob * 100
                else:
                    st.session_state.verdict = "COVER"
                    st.session_state.display_conf = (1 - prob) * 100
                
                st.session_state.scanned = True
                os.remove(tmp_path)

        if st.session_state.scanned:
            if st.session_state.verdict == "STEGO":
                st.error(f"🚨 COMPROMISED DETECTED | {st.session_state.display_conf:.2f}% Confidence")
            else:
                st.success(f"✅ CLEAN | {st.session_state.display_conf:.2f}% Confidence")
            st.progress(st.session_state.prob)

    with tab2:
        if not st.session_state.scanned:
            st.warning("⚠️ Please perform an Initial Scan in the 'Quick Scan' tab first.")
        else:
            st.write("### Calibrated Evidence Extraction")
            st.info(f"**Current Verdict:** {st.session_state.verdict} ({st.session_state.display_conf:.2f}% confidence)")
            
            col_gen, col_dl = st.columns([1, 1])
            with col_gen:
                btn_gen = st.button("GENERATE FORENSIC EVIDENCE")
            
            if btn_gen:
                with st.spinner("Executing Deep Forensic Audit..."):
                    # Use a context manager but copy images to RAM immediately
                    with tempfile.TemporaryDirectory() as img_dir:
                        reporter = AuspexReporter(model, DEVICE, threshold=0.5229)
                        pdf_bytes = reporter.generate_report_bytes(st.session_state.input_tensor, uploaded_file.name, img_dir)
                        st.session_state.pdf_report = pdf_bytes
                        
                        # USE PIL .COPY() TO FORCE DATA INTO RAM BEFORE TEMPDIR DELETES
                        with Image.open(os.path.join(img_dir, "v1.png")) as img: st.session_state.v1 = img.copy()
                        with Image.open(os.path.join(img_dir, "v2.png")) as img: st.session_state.v2 = img.copy()
                        with Image.open(os.path.join(img_dir, "v3.png")) as img: st.session_state.v3 = img.copy()
                        with Image.open(os.path.join(img_dir, "v4.png")) as img: st.session_state.v4 = img.copy()
                        with Image.open(os.path.join(img_dir, "v5.png")) as img: st.session_state.v5 = img.copy()
                        st.session_state.ev_generated = True

            if 'ev_generated' in st.session_state and st.session_state.ev_generated:
                with col_dl:
                    st.download_button(
                        label="📥 DOWNLOAD CALIBRATED FORENSIC REPORT (PDF)",
                        data=st.session_state.pdf_report,
                        file_name=f"Auspex_Audit_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )

                st.write("#### Visual Evidence Panels")
                col_scans = st.columns(2)
                with col_scans[0]:
                    st.image(st.session_state.v1, caption="Bitstream Structure", use_column_width=True)
                    st.image(st.session_state.v3, caption="Neural Decision Hotspots (XAI)", use_column_width=True)
                with col_scans[1]:
                    st.image(st.session_state.v2, caption="Forensic Residual Scan", use_column_width=True)
                    st.image(st.session_state.v4, caption="Temporal Timeline", use_column_width=True)
                
                st.image(st.session_state.v5, caption="Channel Reliance", width=600)