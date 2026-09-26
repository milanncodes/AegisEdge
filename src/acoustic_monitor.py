
import torch
import torch.nn as nn
import numpy as np
import time
import librosa

# --- Constants for Snapdragon AI Hub Compatibility ---
SAMPLE_RATE = 16000
DURATION_SEC = 3
BUFFER_SIZE = SAMPLE_RATE * DURATION_SEC
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
STATIC_INPUT_SHAPE = (1, 1, 64, 188) # (Batch, Channel, Mels, Time)

def compute_log_mel_spectrogram(audio_buffer):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    tensor = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)
    return tensor

class MicroAcousticClassifier(nn.Module):
    def __init__(self):
        super(MicroAcousticClassifier, self).__init__()
        def conv_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU6(inplace=True)
            )
        def dw_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, in_f, 3, stride, 1, groups=in_f, bias=False),
                nn.BatchNorm2d(in_f),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_f, out_f, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU6(inplace=True)
            )
        self.features = nn.Sequential(
            conv_block(1, 32, 2),
            dw_block(32, 64, 1),
            dw_block(64, 128, 2),
            dw_block(128, 128, 1),
            dw_block(128, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class AcousticPipeline:
    def __init__(self):
        self.model = MicroAcousticClassifier()
        self.model.eval()
    def process_live_buffer(self, audio_array):
        start = time.time()
        spec_tensor = compute_log_mel_spectrogram(audio_array)
        with torch.no_grad():
            prob = self.model(spec_tensor).item()
        return prob, (time.time() - start) * 1000

if __name__ == '__main__':
    pipeline = AcousticPipeline()
    ambient = np.random.normal(0, 0.01, BUFFER_SIZE).astype(np.float32)
    t = np.linspace(0, DURATION_SEC, BUFFER_SIZE)
    scream = (np.random.normal(0, 0.02, BUFFER_SIZE) + 0.8 * np.sin(2 * np.pi * 3000 * t) * np.exp(-((t-1.5)**2)/0.01)).astype(np.float32)

    for label, buf in [("Ambient", ambient), ("Distress", scream)]:
        p, l = pipeline.process_live_buffer(buf)
        print(f"{label} -> Prob: {p:.4f} | Latency: {l:.2f}ms")