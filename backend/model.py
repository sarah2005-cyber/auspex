import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Union

# Make sure we can import the model components from the repo root
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the real building blocks from the repository
try:
    from AttentionFeatureFusionSPM import AttentionFeatureFusionSPM
    from LightweightCNNClassifier import LightweightCNNClassifier
    from EmbeddingRateEstimator import EmbeddingRateEstimator
    from ResidualUnit import DualStreamSResNet5Res
except Exception as e:
    # If imports fail, raise a helpful message when used
    # but allow the module to be imported for editing.
    AttentionFeatureFusionSPM = None
    LightweightCNNClassifier = None
    EmbeddingRateEstimator = None
    DualStreamSResNet5Res = None
    _IMPORT_ERROR = e


class FullPipelineModel(nn.Module):
    """Replicate the training-time FullPipelineModel from repo so checkpoints
    load into matching keys. The model composes:
      - DualStreamSResNet5Res -> produces (batch, 48)
      - AttentionFeatureFusionSPM -> (48 -> 64)
      - LightweightCNNClassifier -> classifier head (64 -> logits)
      - EmbeddingRateEstimator -> embedding rate head (64 -> 1)

    This class mirrors the repo's implementation and will accept two inputs
    in forward(spec1024, spec512). For inference convenience, helper
    functions below will also accept a 48- or 64-dimensional feature vector.
    """

    def __init__(self):
        super().__init__()
        if AttentionFeatureFusionSPM is None or LightweightCNNClassifier is None or DualStreamSResNet5Res is None:
            raise ImportError(f"Missing model component imports: {_IMPORT_ERROR}")

        self.dual_stream = DualStreamSResNet5Res()
        self.fusion = AttentionFeatureFusionSPM()
        self.classifier = LightweightCNNClassifier()
        self.embed_estimator = EmbeddingRateEstimator()

    def forward(self, spec1024: torch.Tensor, spec512: torch.Tensor):
        feat48 = self.dual_stream(spec1024, spec512)  # (batch, 48)
        fused64, alpha = self.fusion(feat48)          # (batch,64), (batch,1)
        logits = self.classifier(fused64)             # (batch, 2)
        emb_rate = self.embed_estimator(fused64)      # (batch,1)
        return logits, alpha, emb_rate


def load_model(checkpoint_path: str, device: str = None) -> FullPipelineModel:
    """Load checkpoint into FullPipelineModel. Uses strict=False to be tolerant
    to minor naming differences; for production replace with exact matching class
    definitions and strict=True.
    """
    if device is None:
        device = "cpu"

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return None

    model = FullPipelineModel()
    model.to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    # Support both raw state_dict and training dict wrappers
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint into FullPipelineModel (strict=False).")
    except Exception as e:
        print(f"Warning: load_state_dict failed: {e}")
    model.eval()
    return model


def preprocess_input(raw_features: Union[List[float], dict]):
    """Preprocessing helper that accepts either:
      - a list/1D array of floats representing fused 64-dim features,
      - a list/1D array of 48-dim features (dual-stream output), or
      - a dict with keys 'spec1024' and 'spec512' containing 2D arrays for the two spectrogram inputs.

    Returns a numpy array shaped (batch, features) ready for model_predict_numpy.
    """
    # spec inputs as dict
    if isinstance(raw_features, dict):
        # Expect raw_features = {'spec1024': np.array(...), 'spec512': np.array(...)}
        return raw_features

    x = np.array(raw_features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def model_predict_numpy(model: FullPipelineModel, X: np.ndarray):
    """Run model inference from numpy input X.

    Behavior:
      - If X is a dict with 'spec1024' and 'spec512', run full model forward and
        return logits (as numpy), alpha and emb_rate.
      - If X has shape (batch, 64) -> treat as fused features and run classifier + embed_estimator.
      - If X has shape (batch, 48) -> treat as dual-stream output, run fusion -> classifier.

    Returns a dict with keys depending on path: e.g. {'logits':..., 'alpha':..., 'emb_rate':...}
    """
    model.eval()
    with torch.no_grad():
        # Full spec inputs path
        if isinstance(X, dict):
            spec1024 = torch.from_numpy(np.array(X['spec1024'])).float()
            spec512 = torch.from_numpy(np.array(X['spec512'])).float()
            logits, alpha, emb_rate = model(spec1024, spec512)
            return {
                'logits': logits.detach().cpu().numpy(),
                'alpha': alpha.detach().cpu().numpy(),
                'emb_rate': emb_rate.detach().cpu().numpy()
            }

        t = torch.from_numpy(X).float()
        # fused 64-dim path
        if t.dim() == 1:
            t = t.unsqueeze(0)

        if t.size(1) == 64:
            logits = model.classifier(t)
            emb_rate = model.embed_estimator(t)
            return {
                'logits': logits.detach().cpu().numpy(),
                'emb_rate': emb_rate.detach().cpu().numpy()
            }

        # dual-stream 48-dim path -> fusion -> classifier
        if t.size(1) == 48:
            fused, alpha = model.fusion(t)
            logits = model.classifier(fused)
            emb_rate = model.embed_estimator(fused)
            return {
                'logits': logits.detach().cpu().numpy(),
                'alpha': alpha.detach().cpu().numpy(),
                'emb_rate': emb_rate.detach().cpu().numpy()
            }

        raise ValueError(f"Unexpected input shape for prediction: {t.shape}. Expected 48 or 64 features or a dict with spec1024/spec512.")
