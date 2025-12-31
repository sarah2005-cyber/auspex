import numpy as np
import io
import soundfile as sf
import model
import librosa
import torch
import matplotlib.pyplot as plt

def make_tone(duration_s=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    x = 0.2 * np.sin(2 * np.pi * freq * t)
    return x, sr

def wav_bytes(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.getvalue()

def main():
    # Load model from backend checkpoint
    ckpt = 'FullPipelineModel_seed42_epoch29_F10.8015_best.pt'
    m = model.load_model(ckpt)
    if m is None:
        print('Model did not load')
        return

    x, sr = make_tone()
    b = wav_bytes(x, sr)

    # Load via librosa from bytes
    audio, _ = librosa.load(io.BytesIO(b), sr=22050, mono=True)
    S1024 = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    S512 = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
    S1024 = np.log1p(S1024)
    S512 = np.log1p(S512)

    spec1024 = np.expand_dims(np.expand_dims(S1024, axis=0), axis=0).astype(float)
    spec512 = np.expand_dims(np.expand_dims(S512, axis=0), axis=0).astype(float)

    res = model.model_predict_numpy(m, {'spec1024': spec1024, 'spec512': spec512})
    print('Prediction result keys:', res.keys())
    print('Logits:', res.get('logits'))
    print('Alpha:', res.get('alpha'))
    print('Emb_rate:', res.get('emb_rate'))

    # SHAP on fused features
    with torch.no_grad():
        s1024 = torch.from_numpy(spec1024).float()
        s512 = torch.from_numpy(spec512).float()
        feat48 = m.dual_stream(s1024, s512)
        fused64, alpha = m.fusion(feat48)
        fused_np = fused64.detach().cpu().numpy()

    import shap
    background = np.zeros((1, fused_np.shape[1]), dtype=float)

    def f(X):
        r = model.model_predict_numpy(m, X)
        if 'logits' in r:
            logits = np.array(r['logits'])
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            return probs[:, 1]
        if 'emb_rate' in r:
            return np.array(r['emb_rate']).reshape(-1)
        raise RuntimeError('No logits or emb_rate')

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(fused_np, nsamples=100)
    print('SHAP values shape/type:', type(shap_values))

    # Save a quick plot
    vals = np.array(shap_values)[0] if isinstance(shap_values, list) else np.array(shap_values)
    mean_abs = np.abs(vals).mean(axis=0) if vals.ndim == 2 else np.abs(vals)
    feat_names = [f'f{i}' for i in range(mean_abs.shape[0])]
    plt.figure(figsize=(6,4))
    plt.barh(np.arange(len(feat_names)), mean_abs)
    plt.yticks(np.arange(len(feat_names)), feat_names)
    plt.tight_layout()
    plt.savefig('static/test_shap.png')
    print('Saved SHAP plot to backend/static/test_shap.png')

if __name__ == '__main__':
    main()
