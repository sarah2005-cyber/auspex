from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
import tempfile
import os
import subprocess
from fastapi import HTTPException as _HTTPException
import numpy as np
import librosa
from typing import Any
from preprocess import process_audio_bytes
import traceback

# Import model helpers from local module
try:
    from model import load_model, preprocess_input, model_predict_numpy
except Exception:
    # If relative import fails when running as a module, try package-style
    from .model import load_model, preprocess_input, model_predict_numpy

# FastAPI app and static directory
app = FastAPI()
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Simple input schema for JSON feature requests
class InputData(BaseModel):
    features: Any

# Allow frontend dev server origin by default (adjust as needed)
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files so SHAP plots can be served
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _robust_load_audio_from_bytes(contents: bytes, filename: str = None, sr: int = 22050):
    """Try to load audio from raw bytes. Strategy (in order):
    1) Try librosa directly (soundfile/libsndfile) on the uploaded temp file.
    2) Try transcoding with FFmpeg CLI to WAV (most robust for many codecs).
    3) Try pydub (which also uses FFmpeg) as a fallback.
    If all fail, raise an HTTPException with actionable advice.
    Note: some codecs (e.g., G.729) require ffmpeg built with the appropriate decoder
    or extra libraries (bcg729). If you need support for such codecs, install an
    FFmpeg build with that codec or transcode files to WAV externally.
    """
    suffix = os.path.splitext(filename)[1] if filename else '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 1) Try librosa (soundfile) directly
        try:
            audio, sr_ret = librosa.load(tmp_path, sr=sr, mono=True)
            return audio, sr_ret
        except Exception as e1:
            # 2) Try ffmpeg CLI to transcode to WAV
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
                    tmp2_path = tmp2.name

                # Build ffmpeg command to convert input file to 1-channel WAV at target sample rate
                cmd = [
                    'ffmpeg', '-y', '-nostdin', '-loglevel', 'error',
                    '-i', tmp_path,
                    '-ar', str(sr), '-ac', '1', tmp2_path
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    try:
                        audio, sr_ret = librosa.load(tmp2_path, sr=sr, mono=True)
                        return audio, sr_ret
                    finally:
                        try:
                            os.unlink(tmp2_path)
                        except Exception:
                            pass
                else:
                    ff_err = proc.stderr.strip()
                    # If ffmpeg couldn't decode (common for proprietary codecs), try pydub fallback
                    # but include ffmpeg stderr in any eventual error message for debugging.
            except FileNotFoundError:
                ff_err = 'ffmpeg executable not found on PATH.'

            # 3) Try pydub (which itself calls ffmpeg) as another route
            try:
                from pydub import AudioSegment
            except Exception:
                raise _HTTPException(status_code=400, detail=(
                    f"Could not read uploaded audio. librosa error: {e1}; ffmpeg stderr: {ff_err if 'ff_err' in locals() else '<none>'}. "
                    "Install ffmpeg and pydub, or convert the file to WAV/FLAC before uploading. "
                    "For special codecs like G.729 you may need an FFmpeg build with that codec (e.g. bcg729)."))

            try:
                audio_seg = AudioSegment.from_file(tmp_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp3:
                    tmp3_path = tmp3.name
                audio_seg.export(tmp3_path, format='wav')
                try:
                    audio, sr_ret = librosa.load(tmp3_path, sr=sr, mono=True)
                    return audio, sr_ret
                finally:
                    try:
                        os.unlink(tmp3_path)
                    except Exception:
                        pass
            except Exception as e3:
                raise _HTTPException(status_code=400, detail=(
                    f"Could not decode uploaded audio. librosa error: {e1}; ffmpeg stderr: {ff_err if 'ff_err' in locals() else '<none>'}; pydub error: {e3}. "
                    "Install FFmpeg (with codec support) or transcode files to WAV before uploading."))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
CKPT_PATH = Path(__file__).resolve().parent / "FullPipelineModel_seed42_epoch29_F10.8015_best.pt"


def get_model():
    # Lazy load model and cache on module level
    global _MODEL
    try:
        _MODEL
    except NameError:
        _MODEL = None

    if _MODEL is None:
        _MODEL = load_model(str(CKPT_PATH))
    return _MODEL


@app.post("/predict")
def predict(data: InputData):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    x = preprocess_input(data.features)
    try:
        res = model_predict_numpy(model, x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Prepare friendly JSON output: logits, probabilities (for binary), emb_rate, alpha
    out = {}
    if 'logits' in res:
        logits = np.array(res['logits'])
        out['logits'] = logits.tolist()
        # compute softmax probs for binary / multi-class
        try:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            out['probs'] = probs.tolist()
        except Exception:
            # fallback
            out['probs'] = None

    if 'emb_rate' in res:
        out['emb_rate'] = np.array(res['emb_rate']).tolist()
    if 'alpha' in res:
        out['alpha'] = np.array(res['alpha']).tolist()

    return out


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """Accepts an audio file upload (wav, mp3, etc.), computes spectrograms with
    n_fft=1024 and n_fft=512, and runs the full model forward.
    Returns logits/probs/alpha/emb_rate as JSON.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    contents = await file.read()
    # Use preprocessing pipeline to convert/standardize and compute SPM spectrograms
    try:
        spm = process_audio_bytes(contents, file.filename, out_dir=str(STATIC_DIR))
    except Exception as e:
        # Log full traceback for debugging in server logs, then return a concise HTTP error
        traceback.print_exc()
        print(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

    # Use the SPM-modified spectrograms for model input
    S1024 = spm['spec_1024_spm']
    S512 = spm['spec_512_spm']

    # Prepare tensors: expected shape (batch, 1, freq_bins, frames)
    spec1024 = np.expand_dims(np.expand_dims(S1024, axis=0), axis=0).astype(float)
    spec512 = np.expand_dims(np.expand_dims(S512, axis=0), axis=0).astype(float)

    try:
        res = model_predict_numpy(model, {'spec1024': spec1024, 'spec512': spec512})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    out = {}
    if 'logits' in res:
        out['logits'] = np.array(res['logits']).tolist()
        try:
            logits = np.array(res['logits'])
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            out['probs'] = probs.tolist()
        except Exception:
            out['probs'] = None
    if 'emb_rate' in res:
        out['emb_rate'] = np.array(res['emb_rate']).tolist()
    if 'alpha' in res:
        out['alpha'] = np.array(res['alpha']).tolist()

    return out


@app.post("/explain_file")
async def explain_file(file: UploadFile = File(...)):
    """Accepts an audio file upload and computes SHAP explanations for the
    prediction. Returns shap_values and a saved plot URL.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    contents = await file.read()
    try:
        spm = process_audio_bytes(contents, file.filename, out_dir=str(STATIC_DIR))
    except Exception as e:
        traceback.print_exc()
        print(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

    S1024 = spm['spec_1024_spm']
    S512 = spm['spec_512_spm']

    spec1024 = np.expand_dims(np.expand_dims(S1024, axis=0), axis=0).astype(float)
    spec512 = np.expand_dims(np.expand_dims(S512, axis=0), axis=0).astype(float)

    # For SHAP we explain the flattened fused features if possible: compute model dual_stream -> fusion output
    try:
        # Run model to get fused features by using model.dual_stream and model.fusion
        with torch.no_grad():
            s1024 = torch.from_numpy(spec1024).float()
            s512 = torch.from_numpy(spec512).float()
            feat48 = model.dual_stream(s1024, s512)  # (batch,48)
            fused64, alpha = model.fusion(feat48)
            fused_np = fused64.detach().cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute fused features for SHAP: {e}")

    # Use KernelExplainer on the fused64 vector (shape (1,64))
    try:
        import shap
        import matplotlib.pyplot as plt

        background = np.zeros((1, fused_np.shape[1]), dtype=float)

        def f(X: np.ndarray) -> np.ndarray:
            r = model_predict_numpy(model, X)
            if 'logits' in r:
                logits = np.array(r['logits'])
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                return probs[:, 1]
            if 'emb_rate' in r:
                return np.array(r['emb_rate']).reshape(-1)
            raise RuntimeError('Model prediction did not return logits or emb_rate')

        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(fused_np, nsamples=100)

        plot_path = STATIC_DIR / "shap_plot.png"
        plt.figure(figsize=(6, 4))
        if isinstance(shap_values, list):
            vals = np.array(shap_values)[0]
        else:
            vals = np.array(shap_values)
        mean_abs = np.abs(vals).mean(axis=0) if vals.ndim == 2 else np.abs(vals)
        feat_names = [f"f{i}" for i in range(mean_abs.shape[0])]
        y_pos = np.arange(len(feat_names))
        plt.barh(y_pos, mean_abs, align='center')
        plt.yticks(y_pos, feat_names)
        plt.xlabel('mean |SHAP value|')
        plt.title('SHAP feature importances (fused features)')
        plt.tight_layout()
        plt.savefig(str(plot_path))
        plt.close()

        def to_list(x):
            try:
                return np.array(x).tolist()
            except Exception:
                return x

        return {"shap_values": to_list(shap_values), "plot_url": f"/static/{plot_path.name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explain failed: {e}")


@app.post("/explain")
def explain(data: InputData):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    x = preprocess_input(data.features)

    # Ensure we have a 2D numpy sample for KernelExplainer; for dict inputs SHAP is not supported here
    if isinstance(x, dict):
        raise HTTPException(status_code=400, detail="SHAP /explain currently only supports 48- or 64-dim flat inputs, not full spectrogram dicts.")

    x_np = np.array(x)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)

    # Create a tiny background dataset placeholder (user should replace with real background later)
    background = np.zeros((1, x_np.shape[1]), dtype=float)

    try:
        import shap
        import matplotlib.pyplot as plt

        # wrapper function for SHAP: returns probability for class 1 (positive class)
        def f(X: np.ndarray) -> np.ndarray:
            # model_predict_numpy returns dict; extract logits and compute softmax prob for class 1
            r = model_predict_numpy(model, X)
            if 'logits' in r:
                logits = np.array(r['logits'])
                # compute softmax and return column for class 1
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                # return 1D array of prob(class 1)
                return probs[:, 1]
            # if logits missing, try emb_rate
            if 'emb_rate' in r:
                return np.array(r['emb_rate']).reshape(-1)
            # fallback: raise
            raise RuntimeError('Model prediction did not return logits or emb_rate')

        explainer = shap.KernelExplainer(f, background)

        # KernelExplainer requires 2D array; pass our single sample
        shap_values = explainer.shap_values(x_np, nsamples=100)

        # Try to create a summary/bar plot and save it
        plot_path = STATIC_DIR / "shap_plot.png"
        plt.figure(figsize=(6, 4))
        try:
            # shap_values may be list (for multi-output) or array
            if isinstance(shap_values, list):
                vals = np.array(shap_values)[0]
            else:
                vals = np.array(shap_values)

            # Attempt a simple bar plot of absolute SHAP values for this single instance
            mean_abs = np.abs(vals).mean(axis=0) if vals.ndim == 2 else np.abs(vals)
            feat_names = [f"f{i}" for i in range(mean_abs.shape[0])]
            y_pos = np.arange(len(feat_names))
            plt.barh(y_pos, mean_abs, align='center')
            plt.yticks(y_pos, feat_names)
            plt.xlabel('mean |SHAP value|')
            plt.title('SHAP feature importances')
            plt.tight_layout()
            plt.savefig(str(plot_path))
            plt.close()
        except Exception:
            # Fallback: create text file with shap values
            plot_path = STATIC_DIR / "shap_values.txt"
            with open(plot_path, "w") as fh:
                fh.write(str(shap_values))

        # Return shap values as JSON-friendly list
        def to_list(x):
            try:
                return np.array(x).tolist()
            except Exception:
                return x

        return {"shap_values": to_list(shap_values), "plot_url": f"/static/{plot_path.name}"}

    except Exception as e:
        # If SHAP not installed or explainer fails, return an error with details
        raise HTTPException(status_code=500, detail=f"SHAP explain failed: {e}")
