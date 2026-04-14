import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import time
from datetime import datetime
from scipy.stats import spearmanr, ttest_rel
from captum.attr import IntegratedGradients, Occlusion
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF
import tempfile
from pathlib import Path
from PIL import Image

# ============================================================================
# 1. UPDATED MODEL ARCHITECTURE (StegoCNN - 9 Channel Core)
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G729A_BIT_MAP = {
    "LSP_L0": range(0, 1), "LSP_L1": range(1, 8), "LSP_L2": range(8, 13), "LSP_L3": range(13, 18),
    "Adaptive_CB_Index_P1": range(18, 26), "Parity_P0": range(26, 27), "Fixed_CB_Index_C1": range(27, 40),
    "Fixed_CB_Signs_S1": range(40, 44), "Pitch_Gain_GA1": range(44, 47), "Fixed_CB_Gain_GB1": range(47, 50),
    "Adaptive_CB_Index_P2": range(50, 55), "Fixed_CB_Index_C2": range(55, 68), "Fixed_CB_Signs_S2": range(68, 72),
    "Pitch_Gain_GA2": range(72, 75), "Fixed_CB_Gain_GB2": range(75, 78), "Unused_Sync": range(78, 80)
}

class DualPathwayForensicLayer(nn.Module):
    """
    Dual-Pathway Preprocessing Block:
    1. Fixed Pathway: Deterministic high-pass filters to prevent signal smoothing.
    2. Learnable Pathway: Dynamic filters to catch unforeseen stego traces.
    """
    def __init__(self):
        super().__init__()

        # --- FIXED PATHWAY (Deterministic Truths) ---
        # 1st Order (Sudden flips)
        hp1 = torch.tensor([[[[-1., 1.]]]], dtype=torch.float32)
        # 2nd Order (Local curvature)
        hp2 = torch.tensor([[[[-1., 2., -1.]]]], dtype=torch.float32)
        # 3rd Order (Complex sequential flips)
        hp3 = torch.tensor([[[[-1., 3., -3., 1.]]]], dtype=torch.float32)
        # 3x3 Spatial Laplacian (Cross-dimensional anomalies)
        hp_spatial = torch.tensor([[[[-1., -1., -1.],
                                     [-1.,  8., -1.],
                                     [-1., -1., -1.]]]], dtype=torch.float32)

        self.register_buffer('kernel1', hp1)
        self.register_buffer('kernel2', hp2)
        self.register_buffer('kernel3', hp3)
        self.register_buffer('kernel_spatial', hp_spatial)

        # --- LEARNABLE PATHWAY (Dynamic Features) ---
        # Extracts 2 learnable channels from the raw bits
        self.learnable_filters = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        # Initialize with small random weights to encourage distinct feature learning
        nn.init.xavier_normal_(self.learnable_filters.weight)

    def forward(self, x):
        # Apply residuals to Channel 1 (Raw Bits)
        raw_bits = x[:, 0:1, :, :]

        # 1. Apply Fixed Kernels
        res1 = F.conv2d(raw_bits, self.kernel1, padding=(0, 0))
        res2 = F.conv2d(raw_bits, self.kernel2, padding=(0, 0))
        res3 = F.conv2d(raw_bits, self.kernel3, padding=(0, 0))
        res_sp = F.conv2d(raw_bits, self.kernel_spatial, padding=1) # 3x3 needs padding=1

        # Pad 1D residuals to match original (100, 80) dimensions
        res1 = F.pad(res1, (0, 1, 0, 0))
        res2 = F.pad(res2, (1, 1, 0, 0))
        res3 = F.pad(res3, (1, 2, 0, 0)) # Pad to offset kernel size 4

        # 2. Apply Learnable Kernels
        learned_res = self.learnable_filters(raw_bits)

        # Return all 6 generated channels (4 fixed + 2 learned)
        return torch.cat([res1, res2, res3, res_sp, learned_res], dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# --- Lightweight Channel Attention (SE Block) ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to recalibrate channel saliency."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Generates the XAI Heatmap for review."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn_weights = self.sigmoid(out)
        return x * attn_weights, attn_weights

class StegoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hp_layer = DualPathwayForensicLayer()

        # Input is now 9 channels:
        # (Original 3: Raw, Temp-Diff, Stability) + (Fixed 4) + (Learnable 2) = 9
        self.init_conv = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dw_conv = DepthwiseSeparableConv(32, 64, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.attention = SpatialAttention()

        self.res_block1 = ResidualBlock(64, dropout=0.1)
        self.se1 = SEBlock(64, reduction=16)
        self.res_block2 = ResidualBlock(64, dropout=0.1)
        self.se2 = SEBlock(64, reduction=16)

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 1. Extract Dual-Pathway Residuals (Extracts 6 channels from raw bits)
        hp_features = self.hp_layer(x)

        # 2. Concatenate with original input
        # [Batch, 3, 100, 80] + [Batch, 6, 100, 80] -> [Batch, 9, 100, 80]
        x = torch.cat([x, hp_features], dim=1)

        # 3. Pass the 9-channel tensor into the pipeline
        x = self.init_conv(x)
        x = self.pool2(self.dw_conv(x))
        x, attn_weights = self.attention(x)

        x = self.res_block1(x)
        x = self.se1(x)
        x = self.res_block2(x)
        x = self.se2(x)

        mean_x = torch.mean(x, dim=[2, 3])
        std_x = torch.std(x, dim=[2, 3])
        x = torch.cat([mean_x, std_x], dim=1)

        out = self.fc(x)
        return out, attn_weights

# ============================================================================
# 2. PREPROCESSING
# ============================================================================

def preprocess_g729a(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()[:1000]
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).reshape(-1, 80).astype(np.float32)
    bits = bits[:100, :]
    diff = np.zeros_like(bits); diff[1:] = np.abs(bits[1:] - bits[:-1])
    stability = np.abs(bits - np.mean(bits, axis=0, keepdims=True))
    return torch.from_numpy(np.stack([bits, diff, stability], axis=0)).unsqueeze(0).to(DEVICE)

# ============================================================================
# 3. INTEGRATED RIGOROUS AUSPEX REPORTER
# ============================================================================

class AuspexReporter:
    def __init__(self, model, device, threshold=0.5229):
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.ig = IntegratedGradients(lambda x: self.model(x)[0])
        self.channel_names = ['Raw Bits', 'Temp. Diff', 'Bit Stability']
        self.img_w = 130
        self.small_img_w = 90
        self.center_x = (210 - self.img_w) / 2
        self.center_x_small = (210 - self.small_img_w) / 2

    def _get_file_hash(self, file_buffer):
        hasher = hashlib.md5()
        try:
            file_buffer.seek(0)
            for chunk in iter(lambda: file_buffer.read(4096), b""): 
                hasher.update(chunk)
            file_buffer.seek(0) # Reset again for other functions
            return hasher.hexdigest()
        except Exception as e: 
            return f"HASH_ERROR: {str(e)}"

    def _compute_bit_flip_impact(self, tensor, top_k_percent=0.02):
        """Optimized Causal Analysis: Only flip bits identified as 'important' by XAI."""
        self.model.eval()
        x = tensor.clone().to(self.device)
    
        # Get base probability
        with torch.no_grad():
            base_prob = torch.sigmoid(self.model(x)[0]).item()

        # Identify top indices based on a quick gradient pass or existing attribution
        flat_attr = np.abs(self.ig.attribute(x, target=0, n_steps=5).cpu().numpy().flatten())
        top_indices = np.argsort(flat_attr)[-int(len(flat_attr) * top_k_percent):]

        impact_map = np.zeros(tensor.squeeze(0).shape)
        flat_impact = impact_map.flatten()

        for idx in top_indices:
            x_flip = x.clone()
            x_flip.view(-1)[idx] = 1 - x_flip.view(-1)[idx]
            with torch.no_grad():
                new_prob = torch.sigmoid(self.model(x_flip)[0]).item()
                flat_impact[idx] = np.abs(new_prob - base_prob)

        return flat_impact.reshape(tensor.squeeze(0).shape)

    def _evaluate_insertion(self, tensor, attr, steps=20):
        """Rigorous Check: Starting from a blank signal, does adding 'guilty' bits restore confidence?"""
        idx = np.argsort(-np.abs(attr).flatten())
        step_sz = len(idx) // steps
        x_orig = tensor.clone().to(self.device).contiguous()
        x_base = torch.full_like(x_orig, x_orig.mean()) # Baseline (neutral)

        insertion_scores = []
        for i in range(steps + 1):
            x_ins = x_base.clone()
            x_ins.view(-1)[idx[:i*step_sz]] = x_orig.view(-1)[idx[:i*step_sz]]
            with torch.no_grad():
                insertion_scores.append(torch.sigmoid(self.model(x_ins)[0]).item())
        return insertion_scores

    def _get_topk_sensitivity(self, tensor, attr, k=0.1):
        """Concentration Check: Does the score hold if we ONLY keep top bits?"""
        x = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
        flat_attr = attr.flatten()
        idx = np.argsort(-np.abs(flat_attr))[:int(len(flat_attr) * k)]

        x_top = torch.full_like(x, x.mean()).to(self.device)
        x_top.view(-1)[idx] = x.view(-1)[idx]
        with torch.no_grad():
            return torch.sigmoid(self.model(x_top)[0]).item()

    def _get_bottomk_sensitivity(self, tensor, attr, k=0.2):
        """Negative Control: Does the score stay stable if we remove 'unimportant' bits?"""
        x = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
        flat_attr = attr.flatten()
        idx = np.argsort(np.abs(flat_attr))[:int(len(flat_attr) * k)]

        x_stable = x.clone()
        x_stable.view(-1)[idx] = x.mean() # Neutralize unimportant bits
        with torch.no_grad():
            return torch.sigmoid(self.model(x_stable)[0]).item()

    def _evaluate_faithfulness(self, tensor, attr, steps=20):
        """Calculates how fast model confidence drops when removing 'guilty' bits."""
        idx = np.argsort(-np.abs(attr).flatten())
        step_sz = len(idx) // steps
        x_orig = tensor.clone().to(self.device).contiguous()

        deletion_scores = []
        for i in range(steps + 1):
            x_del = x_orig.clone()
            # Deletion: Replace top bits with the mean (neutralizing them)
            x_del.view(-1)[idx[:i*step_sz]] = x_orig.mean()
            with torch.no_grad():
                deletion_scores.append(torch.sigmoid(self.model(x_del)[0]).item())
        return deletion_scores

    def _test_robustness(self, tensor, attr_orig, flip_prob=0.01):
        """Measures if the explanation stays the same if we add a tiny bit of noise."""
        flipped = tensor.clone()
        mask = torch.rand_like(flipped) < flip_prob
        flipped[mask] = 1 - flipped[mask]

        attr_noisy = self.ig.attribute(flipped, target=0, n_steps=20).squeeze(0).cpu().detach().numpy()
        stability, _ = spearmanr(attr_orig.flatten(), attr_noisy.flatten())
        return stability

    def _get_codec_interpretation(self, attr):
        """Maps attribution back to G.729a parameters (LSP, Pitch, Codebook)."""
        bit_importance = np.abs(attr).mean(axis=0)
        forensic_summary = {}
        for param, indices in G729A_BIT_MAP.items():
            forensic_summary[param] = float(bit_importance[indices].mean())
        return sorted(forensic_summary.items(), key=lambda x: x[1], reverse=True)

    def generate_report(self, tensor, sample_name, img_dir, raw_file_path=None):
        # 1. Neural Attribution (IG)
        input_t = tensor.clone().requires_grad_(True)
        with torch.no_grad():
            logits, _ = self.model(input_t)
            prob = torch.sigmoid(logits).item()
        
        # Attribution pass
        attr = self.ig.attribute(input_t, target=0, n_steps=50).squeeze(0).cpu().detach().numpy()

        # 2. Optimized Causal Analysis (Bit-Flip)
        bit_flip_map = self._compute_bit_flip_impact(tensor)

        # 3. Generate Visuals
        img_paths = self._create_plots(tensor.squeeze(0), attr, bit_flip_map, img_dir)

        # 4. Rigor Metrics & Logic
        correlation, _ = spearmanr(np.abs(attr).flatten(), bit_flip_map.flatten())
        file_hash = self._get_file_hash(raw_file_path) if raw_file_path else "N/A"

        if prob > (self.threshold + 0.1): status, b_color = "COMPROMISED", (231, 76, 60)
        elif prob > self.threshold: status, b_color = "COMPROMISED", (231, 76, 60)
        elif prob > (self.threshold - 0.1): status, b_color = "INCONCLUSIVE", (241, 196, 15)
        else: status, b_color = "CLEAN", (46, 204, 113)

        pdf = FPDF()

        # --- PAGE 1: HEADER & IDENTIFICATION ---
        pdf.add_page()
        pdf.set_fill_color(30, 35, 45); pdf.rect(0, 0, 210, 40, 'F') # Slightly larger header
        pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", 'B', 18)
        pdf.cell(190, 10, "AUSPEX: AUDIT", ln=True, align='C')
        pdf.set_font("Helvetica", 'I', 9)
        pdf.cell(190, 5, f"Causal Validation Enabled | MD5: {file_hash}", ln=True, align='C')
        pdf.ln(5)

        # Examiner & Methodology Identification
        pdf.set_font("Helvetica", 'B', 10); pdf.set_text_color(200, 200, 200)
        pdf.cell(95, 5, f"LEAD EXAMINER: Sarah Rahim", align='L')
        pdf.cell(95, 5, f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align='R', ln=True)
        pdf.ln(10)

        # [1] Metadata
        pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", 'B', 10); pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 7, " [1] ANALYSIS CONTEXT & METHODOLOGY", ln=True, fill=True); pdf.set_font("Courier", '', 9); pdf.ln(1)
        pdf.cell(190, 5, f" > SAMPLE ID: {sample_name}", ln=True)
        pdf.set_font("Courier", 'B', 9)
        pdf.cell(190, 5, f" > METHOD: AUSPEX-v2 Protocol (Residual-CNN + Integrated Gradients + Causal Bit-Flip)", ln=True)
        stability_score = self._test_robustness(tensor, attr)
        codec_findings = self._get_codec_interpretation(attr)

        # Add to Metadata display
        pdf.cell(190, 5, f" > EXPLAINABILITY FAITHFULNESS (Spearman): {correlation:.4f}", ln=True)
        pdf.cell(190, 5, f" > EXPLANATION STABILITY (Robustness): {stability_score:.4f}", ln=True)

        topk_score = self._get_topk_sensitivity(tensor, attr, k=0.1)
        pdf.cell(190, 5, f" > TOP-10% BIT SENSITIVITY (Concentration): {topk_score:.4f}", ln=True)
        pdf.set_font("Courier", 'B', 9); pdf.cell(190, 5, f" > INITIAL FIELD VERDICT: {status}", ln=True); pdf.ln(2)

        # [2] Risk Gauge
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [2] CALIBRATED RISK ASSESSMENT", ln=True, fill=True); pdf.ln(3)
        pdf.set_x(45); pdf.set_font("Helvetica", 'B', 7)
        pdf.cell(40, 3, "LIKELY CLEAN", align='L'); pdf.cell(40, 3, "UNCERTAIN", align='C'); pdf.cell(40, 3, "HIGH RISK", align='R', ln=True)
        pdf.set_draw_color(200, 200, 200); pdf.rect(45, pdf.get_y(), 120, 4); pdf.set_fill_color(*b_color); pdf.rect(45, pdf.get_y(), 120 * prob, 4, 'F')
        pdf.set_draw_color(0, 0, 0); pdf.set_line_width(0.5); pdf.line(45 + (120 * self.threshold), pdf.get_y()-1, 45 + (120 * self.threshold), pdf.get_y()+5)
        pdf.set_y(pdf.get_y() + 5); pdf.set_font("Helvetica", 'B', 9)
        pdf.cell(190, 5, f"Neural Score: {prob:.4f} | Forensic Status: {status}", align='C', ln=True); pdf.ln(3)

        # [3] Combined Physical Artifact
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [3] PHYSICAL SIGNAL AUDIT (RAW VS RESIDUAL)", ln=True, fill=True); pdf.ln(2)
        pdf.image(img_paths[0], x=10, y=None, w=190) # Combined Raw + Res

        # --- PAGE 2: DECISION EVIDENCE ---
        pdf.add_page(); pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [4] EXPLAINABLE DECISION EVIDENCE", ln=True, fill=True); pdf.ln(3)
        pdf.image(img_paths[1], x=self.center_x, y=None, w=self.img_w) # Focus
        pdf.image(img_paths[2], x=self.center_x, y=None, w=self.img_w) # Timeline
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [5] CHANNEL ATTRIBUTION WEIGHT", ln=True, fill=True); pdf.ln(3)
        pdf.image(img_paths[3], x=self.center_x_small, y=None, w=self.small_img_w)

        # --- NEW PAGE 3: RIGOROUS CAUSAL VALIDATION ---
        pdf.add_page(); pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [6] RIGOROUS CAUSAL VALIDATION (BIT-FLIP ANALYSIS)", ln=True, fill=True); pdf.ln(3)
        pdf.image(img_paths[4], x=self.center_x, y=None, w=self.img_w) # Alignment
        pdf.set_font("Helvetica", 'I', 8); pdf.ln(2)
        pdf.multi_cell(190, 4, "Note: This section compares Neural Influence (IG) with Causal Impact (Bit-Flip). High alignment (bright overlap) confirms that the bits identified by the XAI core are physically responsible for the model's verdict.")

        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [7] CODEC PARAMETER COMPROMISE ANALYSIS", ln=True, fill=True); pdf.ln(3)

        pdf.set_font("Courier", 'B', 9)
        pdf.cell(95, 7, " G.729a Parameter Field", border=1); pdf.cell(95, 7, " Neural Sensitivity (Avg Attr)", border=1, ln=True)

        pdf.set_font("Courier", '', 9)
        for param, importance in codec_findings[:8]: # Show top 8
            pdf.cell(95, 6, f" {param}", border=1)
            pdf.cell(95, 6, f" {importance:.6f}", border=1, ln=True)

        pdf.ln(5)
        pdf.image(img_paths[5], x=self.center_x, y=None, w=self.img_w) # Faithfulness
        pdf.set_font("Helvetica", 'I', 8)
        pdf.multi_cell(190, 4, "Validation: The Deletion Curve shows the drop in model confidence as evidence is removed. A steep drop confirms the model is relying on robust traces rather than noise.")

        # --- PAGE 4: GUIDANCE, LIMITATIONS & VERDICT ---
        pdf.add_page();
        pdf.ln(5); pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [8] ARTIFACT TECHNICAL DETAILS", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Helvetica", '', 8.5)
        details = [
            ("Fig 1 (Scan)", "Comparison of raw bitstream values against high-pass filtered residuals."),
            ("Fig 2 (Focus)", "Spatiotemporal heatmap highlighting bit regions driving the model's primary classification."),
            ("Fig 3 (Weight)", "Quantification of neural sensitivity across raw, differential, and stability channels."),
            ("Fig 4 (Timeline)", "Temporal distribution of saliency across the duration of the audio clip."),
            ("Fig 5 (Channels)", "Sensitivity distribution quantifying the influence of raw, differential, and stability features."),
            ("Fig 6 (Causal)", "Overlay validation comparing neural gradients against physical bit-flip impact results."),
            ("Fig 7 (Faithful)", "Statistical curve proving model confidence drops in direct response to evidence removal.")]
        for t, d in details:
            pdf.set_font("Helvetica", 'B', 8.5); pdf.cell(30, 4, f" {t}:", ln=0); pdf.set_font("Helvetica", '', 8.5); pdf.cell(160, 4, d, ln=True)

        # [8] Pipeline Limitations
        pdf.ln(3); pdf.set_font("Helvetica", 'B', 10); pdf.cell(190, 7, " [9] PIPELINE LIMITATIONS", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Helvetica", 'I', 8)
        lims = ["1. Sensitivity: Detection accuracy degrades at low embedding rates (< 0.1 bpc).",
                "2. Causal Scope: Bit-flip analysis is computationally intensive; limited to core bitstream.",
                "3. Attribution: Heatmaps indicate neural influence, not physical data extraction."]
        for l in lims: pdf.cell(190, 4, f" - {l}", ln=True)

        # [9] Investigative Findings
        pdf.ln(4); pdf.set_font("Helvetica", 'B', 11); pdf.cell(190, 8, " [10] INVESTIGATIVE FINDINGS", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Helvetica", '', 10)
        msg = "Findings: Indicators suggest pattern manipulation consistent with steganography." if prob > self.threshold else "Findings: Signal characteristics remain within natural variance."
        pdf.multi_cell(190, 5, f"{msg} Forensic secondary audit recommended." if prob > self.threshold else f"{msg} No further action required.")

        pdf.ln(10); pdf.set_font("Helvetica", 'B', 15); pdf.set_text_color(*b_color)
        pdf.cell(190, 15, f"!!! FINAL VERDICT: {status} !!!", border=1, ln=True, align='C')

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return pdf_bytes

    def _create_plots(self, tensor, attr, bitflip_map, img_dir):
        fnames = [f"v{i}.png" for i in range(1, 7)]
        paths = [os.path.join(img_dir, f) for f in fnames]

        # Helper for standard plots
        def save_f(path, title, data, xl="", yl="", cmap=None, is_bar=False, sz=(10, 6)):
            plt.figure(figsize=sz)
            if is_bar:
                plt.bar(self.channel_names, data, color=['#5D6D7E', '#EB984E', '#17A589'], edgecolor='black')
                plt.ylabel(yl)
            elif cmap:
                im = plt.imshow(data, cmap=cmap, aspect='auto')
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel(xl); plt.ylabel(yl)
            else:
                plt.fill_between(range(len(data)), data, color='teal', alpha=0.3)
                plt.plot(data, color='teal')
                plt.xlabel(xl); plt.ylabel(yl)
            plt.title(title, fontweight='bold', fontsize=14)
            plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()

        # v1: Combined Physical Scan (Raw + Residual)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(tensor[0].cpu().numpy(), cmap='gray', aspect='auto')
        axes[0].set_title("A. Raw Bitstream Scan", fontweight='bold')
        axes[0].set_xlabel("Bit Offset (0-79)"); axes[0].set_ylabel("Frame Index")
        hp_k = torch.tensor([[[[-1., -1., -1.], [ -1.,  8., -1.], [-1., -1., -1.]]]]).to(self.device)
        res = F.conv2d(tensor.unsqueeze(0)[:, 0:1, :, :], hp_k, padding=1).squeeze().cpu().numpy()
        axes[1].imshow(res, cmap='inferno', aspect='auto')
        axes[1].set_title("B. Residual Scan", fontweight='bold')
        axes[1].set_xlabel("Bit Offset"); axes[1].set_ylabel("Frame Index")

        plt.tight_layout(); plt.savefig(paths[0], dpi=150); plt.close()

        # v2: Neural Focus (XAI)
        plt.figure(figsize=(10, 6))
        plt.imshow(tensor[0].cpu().numpy(), cmap='gray', alpha=0.3, aspect='auto')
        mask = np.abs(attr[0]) > np.percentile(np.abs(attr[0]), 98)
        overlay = np.ma.masked_where(~mask, attr[0])
        plt.imshow(overlay, cmap='cool', interpolation='none', aspect='auto')
        plt.title("Neural Decision Focus (XAI)", fontweight='bold')
        plt.xlabel("Bit Offset (0-79)"); plt.ylabel("Frame Index")
        plt.savefig(paths[1], dpi=120, bbox_inches='tight'); plt.close()

        # v3: Temporal Timeline
        save_f(paths[2], "Temporal Attribution Timeline", np.abs(attr).sum(axis=(0, 2)), xl="Frame Index", yl="Magnitude")

        # v4: Channel Influence
        imp = np.abs(attr).sum(axis=(1, 2))
        save_f(paths[3], "Attribution Intensity by Channel", (imp/imp.sum())*100, yl="Influence (%)", is_bar=True, sz=(7, 4))

        # v5: Causal Alignment
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        h_ig = np.abs(attr).sum(axis=0); h_ig = (h_ig - h_ig.min()) / (h_ig.max() - h_ig.min() + 1e-8)
        h_bf = np.abs(bitflip_map).sum(axis=0); h_bf = (h_bf - h_bf.min()) / (h_bf.max() - h_bf.min() + 1e-8)
        titles = ["Neural IG Influence", "Causal Bit-Flip Impact", "Alignment Overlay"]
        for i, data in enumerate([h_ig, h_bf, None]):
            ax = axes[i]
            if i < 2:
                ax.imshow(data, cmap='hot' if i==0 else 'viridis', aspect='auto')
            else:
                ax.imshow(h_ig, cmap='hot', alpha=0.7, aspect='auto')
                ax.imshow(h_bf, cmap='viridis', alpha=0.3, aspect='auto')
            ax.set_title(titles[i], fontweight='bold')
            ax.set_xlabel("Bit Index"); ax.set_ylabel("Frame")
            for param, indices in G729A_BIT_MAP.items():
                ax.axvline(x=indices[0], color='cyan', linestyle='--', alpha=0.1)
        plt.tight_layout(); plt.savefig(paths[4], dpi=150); plt.close()

        # v6: Faithfulness Curves
        del_scores = self._evaluate_faithfulness(tensor.unsqueeze(0), attr)
        ins_scores = self._evaluate_insertion(tensor.unsqueeze(0), attr)
        plt.figure(figsize=(10, 5))
        plt.plot(del_scores, label="Deletion (Evidence Removal)", color='#C0392B', marker='o')
        plt.plot(ins_scores, label="Insertion (Evidence Recovery)", color='#27AE60', marker='s')
        plt.title("7. Forensic Faithfulness: Dual-Curve Analysis", fontweight='bold')
        plt.xlabel("% of Top-Ranked Bits Affected (Evidence Ranking)"); plt.ylabel("Model Probability (Confidence)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(paths[5], dpi=150); plt.close()

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

st.title("🛡️ AUSPEX: Dashboard")
st.warning("This system is intended for decision support in forensic and cybersecurity analysis. Outputs are assistive and should not be treated as definitive legal proof.")

st.markdown("""
### Workflow
1. Upload a G.729a bitstream
2. Run Initial Scan
3. Review verdict
4. Generate forensic supportive evidence
5. Download PDF report
""")

@st.cache_resource
def load_model(path):
    model = StegoCNN()
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        return model.eval().to(DEVICE)
    return None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "Auspex_Forensic_Final_Original_3SeedRun_seed42_best.pt")
model_path = st.sidebar.text_input("Model Checkpoint", "Loaded (internal)")
debug_mode = st.sidebar.checkbox("Enable NFR Debug Info", value=True)
uploaded_file = st.file_uploader("Upload G.729a Bitstream", type=["g729a", "npy"])

model = load_model(model_path)

if 'scanned' not in st.session_state:
    st.session_state.scanned = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = ""
if 'ev_generated' not in st.session_state:
    st.session_state.ev_generated = False
if 'nfr_logs' not in st.session_state:
    st.session_state.nfr_logs = []

def log_nfr_event(event_type, details):
    st.session_state.nfr_logs.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "details": details
    })

if uploaded_file is not None and uploaded_file.size == 0:
    st.error("Uploaded file is empty.")
    log_nfr_event("reliability_error", "Empty file uploaded")

if uploaded_file and model:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.scanned = False
        st.session_state.ev_generated = False
        st.session_state.current_file = uploaded_file.name

    tab1, tab2 = st.tabs(["QUICK SCAN", "DEEP ANALYSIS"])

    with tab1:
        st.write("### Sub-perceptual Integrity Check")
        if st.button("RUN INITIAL SCAN"):
            with st.spinner("Processing..."):
                scan_start = time.perf_counter()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

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
                    st.session_state.ev_generated = False
                    log_nfr_event("reliability_success", f"Quick scan success for {uploaded_file.name}")

                except Exception:
                    st.error("Processing failed due to invalid or unsupported input.")
                    st.session_state.scanned = False
                    log_nfr_event("reliability_error", f"Quick scan failed for {uploaded_file.name}")
                    log_nfr_event("security_safe_error", "Invalid input blocked without stack trace exposure")

                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)

                scan_end = time.perf_counter()
                scan_elapsed = scan_end - scan_start
                st.session_state.scan_time = scan_elapsed
                log_nfr_event("performance_quick_scan", f"{scan_elapsed:.4f}s for {uploaded_file.name}")

                if debug_mode:
                    st.caption(f"Quick Scan Time: {scan_elapsed:.4f} seconds")

        if st.session_state.scanned:
            if st.session_state.verdict == "STEGO":
                st.error(f"COMPROMISED DETECTED | {st.session_state.display_conf:.2f}% Confidence")
            else:
                st.success(f"CLEAN | {st.session_state.display_conf:.2f}% Confidence")
            st.progress(st.session_state.prob)

            if debug_mode:
                st.write({
                    "raw_probability": float(st.session_state.prob),
                    "verdict": st.session_state.verdict,
                    "display_confidence": float(st.session_state.display_conf)
                })

    with tab2:
        if not st.session_state.scanned:
            st.warning("Please perform an Initial Scan in the 'Quick Scan' tab first.")
        else:
            st.write("### Calibrated Evidence Extraction")
            st.info(f"**Current Verdict:** {st.session_state.verdict} ({st.session_state.display_conf:.2f}% confidence)")
            
            col_gen, col_dl = st.columns([1, 1])
            with col_gen:
                btn_gen = st.button("GENERATE FORENSIC SUPPORTIVE EVIDENCE")
            
            if btn_gen:
                with st.spinner("Executing Deep Audit..."):
                    deep_start = time.perf_counter()
                    try:
                        with tempfile.TemporaryDirectory() as img_dir:
                            reporter = AuspexReporter(model, DEVICE, threshold=0.5229)
                            
                            # Generate report and catch the bytes
                            pdf_bytes = reporter.generate_report(
                                st.session_state.input_tensor,
                                uploaded_file.name,
                                img_dir,
                                uploaded_file
                            )
                            st.session_state.pdf_report = pdf_bytes
                            
                            # Load images from img_dir into session_state
                            with Image.open(os.path.join(img_dir, "v1.png")) as img:
                                st.session_state.v1 = img.copy()
                            with Image.open(os.path.join(img_dir, "v2.png")) as img:
                                st.session_state.v2 = img.copy()
                            with Image.open(os.path.join(img_dir, "v3.png")) as img:
                                st.session_state.v3 = img.copy()
                            with Image.open(os.path.join(img_dir, "v4.png")) as img:
                                st.session_state.v4 = img.copy()
                            with Image.open(os.path.join(img_dir, "v5.png")) as img:
                                st.session_state.v5 = img.copy()
                            with Image.open(os.path.join(img_dir, "v6.png")) as img:
                                st.session_state.v6 = img.copy()

                            st.session_state.ev_generated = True
                            log_nfr_event("reliability_success", f"Deep analysis success for {uploaded_file.name}")

                    except Exception:
                        st.error("Deep analysis failed due to invalid input or report generation issue.")
                        st.session_state.ev_generated = False
                        log_nfr_event("reliability_error", f"Deep analysis failed for {uploaded_file.name}")
                        log_nfr_event("security_safe_error", "Deep analysis error handled without exposing stack trace")

                    deep_end = time.perf_counter()
                    deep_elapsed = deep_end - deep_start
                    st.session_state.deep_time = deep_elapsed
                    log_nfr_event("performance_deep_analysis", f"{deep_elapsed:.4f}s for {uploaded_file.name}")

                    if debug_mode:
                        st.caption(f"Deep Analysis Time: {deep_elapsed:.4f} seconds")

            if 'ev_generated' in st.session_state and st.session_state.ev_generated:
                with col_dl:
                    st.download_button(
                        label="DOWNLOAD CALIBRATED FORENSIC SUPPORTIVE REPORT (PDF)",
                        data=st.session_state.pdf_report,
                        file_name=f"Auspex_Audit_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )

                st.write("#### Visual Evidence Panels")
                st.image(st.session_state.v1, caption="Physical Signal Audit (Raw vs Residual)", use_column_width=True)
                
                col_mid = st.columns(2)
                with col_mid[0]:
                    st.image(st.session_state.v2, caption="Neural Decision Focus (XAI)", use_column_width=True)
                with col_mid[1]:
                    st.image(st.session_state.v3, caption="Temporal Attribution Timeline", use_column_width=True)
                
                st.image(st.session_state.v4, caption="Channel Reliance", width=500)
                st.image(st.session_state.v5, caption="Causal Bit-Flip Alignment", use_column_width=True)
                st.image(st.session_state.v6, caption="Forensic Faithfulness Curve", use_column_width=True)

if debug_mode and st.session_state.nfr_logs:
    st.write("### NFR Debug Log")
    st.dataframe(pd.DataFrame(st.session_state.nfr_logs))