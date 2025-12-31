import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import convolve2d

# Spectrogram frame targets
MAX_FRAMES_1024 = 87
MAX_FRAMES_512 = 173


def _ffmpeg_convert_to_wav(input_path: str, output_path: str, sr: int = 44100):
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", input_path,
        "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr}")


def _pydub_convert_to_wav(input_path: str, output_path: str, sr: int = 44100):
    # pydub requires ffmpeg but can be slightly more tolerant for some wrappers
    from pydub import AudioSegment
    seg = AudioSegment.from_file(input_path)
    seg = seg.set_frame_rate(sr).set_channels(1).set_sample_width(2)
    seg.export(output_path, format="wav")


def standardize_audio_from_bytes(contents: bytes, original_filename: str, out_dir: str = None) -> Tuple[str, np.ndarray, int]:
    """Given raw file bytes and the original filename, convert to a standardized
    WAV file (44100 Hz, mono, PCM16) and return (standardized_path, audio_np, sr).
    The standardized file is named `<original_name>_standardized.wav` and saved
    in `out_dir` if provided, otherwise in a temp directory.
    """
    orig_name = Path(original_filename).stem if original_filename else "upload"
    if out_dir:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        out_dir_path = Path(tempfile.mkdtemp())

    standardized_name = f"{orig_name}_standardized.wav"
    standardized_path = str(out_dir_path / standardized_name)

    # write input to temp file preserving extension if possible
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix if original_filename else ".tmp") as tf:
        tf.write(contents)
        tmp_in = tf.name

    try:
        # Prefer ffmpeg conversion if available
        if shutil.which("ffmpeg"):
            try:
                _ffmpeg_convert_to_wav(tmp_in, standardized_path, sr=44100)
            except Exception:
                # fallback to pydub
                _pydub_convert_to_wav(tmp_in, standardized_path, sr=44100)
        else:
            # Try pydub directly (will fail if ffmpeg missing)
            _pydub_convert_to_wav(tmp_in, standardized_path, sr=44100)

        # Load standardized file to numpy
        audio, sr = sf.read(standardized_path, dtype='int16')
        # Convert to float32 in [-1,1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return standardized_path, audio, sr
    finally:
        try:
            os.unlink(tmp_in)
        except Exception:
            pass


def _compute_mag_log_minmax(spec: np.ndarray) -> np.ndarray:
    # spec: complex STFT matrix -> magnitude expected
    mag = np.abs(spec)
    # log10 transform with small eps
    log = np.log10(mag + 1e-8)
    mn = log.min()
    mx = log.max()
    if mx - mn <= 1e-8:
        return np.zeros_like(log)
    norm = (log - mn) / (mx - mn)
    return norm


def _pad_or_trim_time_axis(spec: np.ndarray, max_frames: int) -> np.ndarray:
    # spec shape: (freq_bins, frames)
    freq_bins, frames = spec.shape
    if frames == max_frames:
        return spec
    if frames < max_frames:
        pad_width = max_frames - frames
        padded = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        return padded
    # trim (take first max_frames)
    return spec[:, :max_frames]


KERNEL_STRONG = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
KERNEL_LIGHT = np.array([[-0.5, -0.5, -0.5], [-0.5, 4.0, -0.5], [-0.5, -0.5, -0.5]])


def apply_spm(spec_norm: np.ndarray) -> np.ndarray:
    # spec_norm expected in [0,1], shape (freq_bins, frames)
    conv_strong = convolve2d(spec_norm, KERNEL_STRONG, mode='same', boundary='symm')
    conv_light = convolve2d(spec_norm, KERNEL_LIGHT, mode='same', boundary='symm')
    combined = 1.0 * conv_strong + 0.5 * conv_light
    relu = np.clip(combined, 0.0, None)
    mn = relu.min()
    mx = relu.max()
    if mx - mn <= 1e-8:
        return np.zeros_like(relu)
    return (relu - mn) / (mx - mn)


def compute_spms_from_audio(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Given mono audio at any sr (we will assume it's 44100), compute
    spec1024, spec512, and their SPM versions as specified. Returns dictionary.
    """
    # Ensure audio is sampled at 44100 for consistency; if not, resample
    if sr != 44100:
        audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=44100)
        sr = 44100

    # Compute STFTs with Hamming window
    S1024 = librosa.stft(audio, n_fft=1024, hop_length=512, window='hamming')
    S512 = librosa.stft(audio, n_fft=512, hop_length=256, window='hamming')

    spec1024_norm = _compute_mag_log_minmax(S1024)
    spec512_norm = _compute_mag_log_minmax(S512)

    # Trim/pad time axis
    spec1024_norm = _pad_or_trim_time_axis(spec1024_norm, MAX_FRAMES_1024)
    spec512_norm = _pad_or_trim_time_axis(spec512_norm, MAX_FRAMES_512)

    spec1024_spm = apply_spm(spec1024_norm)
    spec512_spm = apply_spm(spec512_norm)

    return {
        'spec_1024': spec1024_norm,
        'spec_512': spec512_norm,
        'spec_1024_spm': spec1024_spm,
        'spec_512_spm': spec512_spm,
        'frames_1024': spec1024_norm.shape[1],
        'frames_512': spec512_norm.shape[1]
    }


def process_audio_bytes(contents: bytes, original_filename: str, out_dir: str = None) -> Dict:
    """Full pipeline: convert input to standardized WAV (44100Hz, mono, PCM16),
    compute multi-window spectrograms and SPM versions, and return dict with
    paths and numpy arrays.
    """
    standardized_path, audio, sr = standardize_audio_from_bytes(contents, original_filename, out_dir=out_dir)
    spm_dict = compute_spms_from_audio(audio, sr)
    spm_dict['standardized_path'] = standardized_path
    return spm_dict
