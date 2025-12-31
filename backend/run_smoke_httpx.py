import io
import numpy as np
import soundfile as sf
import httpx
import app as backend_app

def make_tone(duration_s=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    x = 0.2 * np.sin(2 * np.pi * freq * t)
    return x, sr

def main():
    x, sr = make_tone()
    buf = io.BytesIO()
    sf.write(buf, x, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)

    with httpx.Client(app=backend_app.app, base_url='http://testserver') as client:
        files = {'file': ('tone.wav', buf, 'audio/wav')}
        print('Posting to /predict_file...')
        r = client.post('/predict_file', files=files, timeout=120.0)
        print('Status:', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception as e:
            print('No JSON:', e, r.text[:500])

        buf.seek(0)
        files = {'file': ('tone.wav', buf, 'audio/wav')}
        print('Posting to /explain_file...')
        r2 = client.post('/explain_file', files=files, timeout=300.0)
        print('Status:', r2.status_code)
        try:
            print('JSON:', r2.json())
        except Exception as e:
            print('No JSON:', e, r2.text[:500])

if __name__ == '__main__':
    main()
