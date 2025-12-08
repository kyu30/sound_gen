from pathlib import Path
import numpy as np
import librosa, soundfile as sf

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
FMIN = 20
FMAX = None
MEL_KW = dict(htk=False, norm="slaney")

def vae_mel_to_wav(npy_path, wav_path,
                   sr=SR, db_lo=-80.0, db_hi=0.0,
                   stretch=True):
    """
    Convert a VAE-generated mel (roughly [0,1]) to audio.
    - npy_path: VAE output .npy file (shape (n_mels, frames) or (1,n_mels,frames))
    - wav_path: where to save the .wav
    """
    mel = np.load(npy_path).astype("float32")

    # squeeze (1,H,W) or (H,W,1) -> (H,W)
    if mel.ndim == 3:
        if mel.shape[0] == 1:
            mel = mel[0]
        elif mel.shape[-1] == 1:
            mel = mel[..., 0]

    # clean and clamp
    mel = np.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)

    # optional: stretch current range [min,max] to [0,1]
    if stretch:
        mn, mx = float(mel.min()), float(mel.max())
        if mx > mn + 1e-8:
            mel01 = (mel - mn) / (mx - mn)
        else:
            mel01 = np.zeros_like(mel)
    else:
        mel01 = np.clip(mel, 0.0, 1.0)

    # [0,1] -> dB
    mel_db = mel01 * (db_hi - db_lo) + db_lo  # e.g. [-80, 0]
    y = mel_db_to_wav(mel_db, sr=sr)
    sf.write(wav_path, y, sr)
    return wav_path


def mel_db_to_wav(M_db, sr=SR):
    M_db = M_db.astype("float32")
    M_db = np.nan_to_num(M_db, nan=-80.0, posinf=0.0, neginf=-80.0)
    M_pow = librosa.db_to_power(M_db).astype("float32")
    S = librosa.feature.inverse.mel_to_stft(
        M_pow, sr=sr, n_fft=N_FFT, fmin=FMIN, fmax=FMAX, **MEL_KW
    )
    y = librosa.griffinlim(S.astype("float32"), n_iter=200,
                           hop_length=HOP, n_fft=N_FFT, momentum=0.99)
    y = y / (abs(y).max() + 1e-8)
    return y
    
for i in range(61):
    if not Path(rf"checkpoints_vq/samples_epoch{i}").exists():
        continue
    else:
        in_dir  = Path(rf"checkpoints_vq/samples_epoch{i}")  # change epoch if needed
        out_dir = in_dir / "wavs"
        out_dir.mkdir(exist_ok=True)

        for p in in_dir.glob("*.npy"):
            wav_path = out_dir / (p.stem + ".wav")
            print("Converting:", p.name, "->", wav_path.name)
            vae_mel_to_wav(str(p), str(wav_path))
        print("Done, check:", out_dir)