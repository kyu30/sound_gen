import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = '..'
out_dir = os.path.join(base_dir, 'spectrograms')
png_out = os.path.join(base_dir, 'pngs')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(png_out, exist_ok=True)
for i in os.listdir(base_dir):
    if i.lower().endswith(".wav"):
        wav_path = os.path.join(base_dir, i)
        y, sr = librosa.load(wav_path, sr = 22050)
        D = librosa.stft(y, n_fft = 1024, hop_length = 256)
        mel_spec = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 1024, hop_length = 256, n_mels = 128)
        mel_db = librosa.power_to_db(mel_spec, ref = np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-Spectrogram")
        plt.tight_layout()
        png_path = os.path.join(png_out, f"{wav_path}_mel.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        npy_path = os.path.join(out_dir, i.replace(".wav","_mel_db.npy"))
        np.save(npy_path, mel_db)
