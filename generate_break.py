from os import replace
import torch
import soundfile as sf
import numpy as np

MODEL_PATH = "drum_model_v2_streaming.ts"   # change if filename differs
OUTPUT_PATH = "breaks_v2\\drum_break.wav"

# Higher = longer break. You can experiment with 256, 512, 1024, etc.
TIME_STEPS = 256  

print("Loading model...")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# ---- latent dimension from model ----
latent_pca = getattr(model, "latent_pca", None)
if latent_pca is None:
    for name, buf in model.named_buffers():
        if "latent_pca" in name:
            latent_pca = buf
            break

if latent_pca is None:
    raise RuntimeError("Could not find latent_pca buffer on model")

latent_dim = latent_pca.shape[0]
print(f"Detected latent_dim = {latent_dim}")

# ---- build z with shape [batch, channels, time] ----
batch_size = 1
channels = latent_dim          # 128 for your model
time_steps = TIME_STEPS

z = torch.randn(batch_size, channels, time_steps)

print("Running model.decode(z)...")
with torch.no_grad():
    y = model.decode(z)    # expected shape: [batch, audio_channels, samples]

# take first batch
audio = y[0].cpu().numpy()

# if multi-channel, mix to mono
if audio.ndim == 2:
    audio = audio.mean(axis=0)

# normalize
audio = audio / (np.max(np.abs(audio)) + 1e-9)

# RAVE models are usually 48kHz
sr = 42000
for i in range(40):
    sf.write(OUTPUT_PATH.replace('break.wav', str(i)+'.wav'), audio, samplerate=sr)

    print(f"Done! Wrote {OUTPUT_PATH.replace('break.wav', str(i)+'.wav')}")
