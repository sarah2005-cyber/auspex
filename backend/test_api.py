from fastapi.testclient import TestClient
from app import app
import numpy as np
import io
import soundfile as sf

client = TestClient(app)

def make_tone(duration_s=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    x = 0.2 * np.sin(2 * np.pi * freq * t)
    return x, sr

def make_wav_bytes(x, sr):
    buf = io.BytesIO()
    # write float32 PCM
    sf.write(buf, x, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf

def main():
    x, sr = make_tone()
    wav_buf = make_wav_bytes(x, sr)

    files = {'file': ('test.wav', wav_buf, 'audio/wav')}

    print('Calling /predict_file...')
    r = client.post('/predict_file', files=files)
    print('Status', r.status_code)
    try:
        print('JSON:', r.json())
    except Exception:
        print('Response text:', r.text)

    # Reset buffer for second request
    wav_buf.seek(0)
    files = {'file': ('test.wav', wav_buf, 'audio/wav')}

    print('\nCalling /explain_file...')
    r2 = client.post('/explain_file', files=files)
    print('Status', r2.status_code)
    try:
        print('JSON:', r2.json())
    except Exception:
        print('Response text:', r2.text)

if __name__ == '__main__':
    main()
