'''
Batch convert .wav files to mel-spectrogram .png images.
'''

import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128

def wav_to_mel_png(wav_path, png_path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )
    M_db = librosa.power_to_db(M, ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        M_db,
        sr=sr,
        hop_length=HOP,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(Path(wav_path).name)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()

if __name__ == "__main__":
    for i in range(61):
        if not Path(rf"checkpoints_vq/samples_epoch{i}").exists():
            continue
        else:
            if __name__ == "__main__":
                in_dir = Path(rf"checkpoints_vq/samples_epoch{i}/wavs")
                out_dir = in_dir / "mel_png"
                out_dir.mkdir(parents=True, exist_ok=True)

                wavs = list(in_dir.glob("*.wav"))
                print("Found", len(wavs), "wav files")
                for w in wavs:
                    out = out_dir / (w.stem + "_mel.png")
                    print("Converting", w.name, "->", out.name)
                    wav_to_mel_png(str(w), str(out))

                print("Done. Images in:", out_dir)
