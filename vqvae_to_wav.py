import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
FMIN = 20
FMAX = None


def vqvae_npy_to_wav(npy_path: Path, wav_path: Path):
    print(f"[info] loading {npy_path.name}")
    mel = np.load(npy_path.as_posix())
    mel = np.array(mel)
    if mel.ndim == 3:
        if mel.shape[0] == 1:
            mel = mel[0]
        elif mel.shape[-1] == 1:
            mel = mel[..., 0]
    if mel.ndim != 2:
        raise ValueError(f"Unexpected mel shape {mel.shape}, expected (128, T) or (1,128,T)")
    mel = mel.astype(np.float32)
    mel = np.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"[info] shape: {mel.shape}, min: {mel.min():.4f}, max: {mel.max():.4f}")
    mel = np.clip(mel, 0.0, 1.0)
    mel_db = mel * 80.0 - 80.0
    mel_power = librosa.db_to_power(mel_db).astype(np.float32)
    S = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        htk=False,
        norm="slaney",
    ).astype(np.float32)
    rng = np.random.default_rng()
    phase = rng.uniform(0.0, 2.0 * np.pi, size=S.shape).astype(np.float32)
    S_complex = S * np.exp(1j * phase)
    y = librosa.istft(
        S_complex,
        hop_length=HOP,
        win_length=N_FFT,
    ).astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)

    sf.write(wav_path.as_posix(), y, SR)
    print(f"[ok] saved {wav_path.name}")


def main():
    in_dir = Path(r"..\..\..\checkpoints_vq\samples_epoch60")
    out_dir = in_dir / "wav_vqvae_randomphase"
    out_dir.mkdir(parents=True, exist_ok=True)

    npys = sorted(in_dir.glob("*.npy"))
    print(f"[info] found {len(npys)} npy files in {in_dir}")
    if not npys:
        return

    for npy in npys:
        out_wav = out_dir / (npy.stem + ".wav")
        try:
            vqvae_npy_to_wav(npy, out_wav)
        except Exception as e:
            print(f"[ERR] {npy.name}: {e}")


if __name__ == "__main__":
    main()
