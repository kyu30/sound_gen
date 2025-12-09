import argparse, csv, json
from pathlib import Path
import numpy as np
import random

def iter_npy(root: Path):
    for path in root.rglob("*.npy"):
        yield path

def running_mean_std(count, mean, M2, new_x):
    x = new_x.astype(np.float64).ravel()
    for xi in x:
        count += 1
        delta = xi - mean
        mean += delta/count
        M2 += delta * (xi - mean)
    return count, mean ,M2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_root", type=str, required=True, help="Folder containing .npy spectrograms")
    ap.add_argument("--out_csv", type=str, default="spectrograms_data.csv")
    ap.add_argument("--out_stats", type=str, default="stats.json")
    ap.add_argument("--sr", type=int, default=22050, help="(Optional) Used to estimate seconds")
    ap.add_argument("--hop_length", type=int, default=256, help="(Optional) Used to estimate seconds")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--compute_stats", action="store_true", help="Compute global mean/std across all values")
    args = ap.parse_args()

    root = Path(args.npy_root).expanduser().resolve()
    files = sorted([p for p in iter_npy(root)])
    if not files:
        raise SystemExit(f"No .npy files found under {root}")

    # Shuffle for split
    rng = random.Random(args.seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * args.train)
    n_val   = int(n * args.val)
    # ensure all assigned
    split_idxs = {
        "train": set(range(0, n_train)),
        "val":   set(range(n_train, n_train + n_val)),
        "test":  set(range(n_train + n_val, n)),
    }

    # Running stats
    count = 0
    mean = 0.0
    M2 = 0.0

    rows = []
    for idx, p in enumerate(files):
        try:
            arr = np.load(p.as_posix(), mmap_mode="r")  # memmap to keep it light
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D mel (n_mels, frames), got shape {arr.shape}")
            n_mels, frames = arr.shape
            seconds = (frames * args.hop_length) / float(args.sr)

            if idx in split_idxs["train"]:
                split = "train"
            elif idx in split_idxs["val"]:
                split = "val"
            else:
                split = "test"

            rows.append({
                "npy_path": str(p).split("Downloads\\")[1],
                "n_mels": n_mels,
                "frames": frames,
                "seconds_est": round(seconds, 6),
                "split": split
            })

            if args.compute_stats:
                count, mean, M2 = running_mean_std(count, mean, M2, arr)

        except Exception as e:
            print(f"SKIP {p} :: {e}")

    # Write CSV
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npy_path","n_mels","frames","seconds_est","split"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV: {out_csv} ({len(rows)} rows)")

    # Write stats (if requested)
    if args.compute_stats and count > 1:
        variance = M2 / (count - 1)
        std = float(np.sqrt(variance))
        stats = {"mean": float(mean), "std": std, "count_values": int(count)}
        out_stats = Path(args.out_stats).resolve()
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats: {out_stats}  (global value mean/std over all spectrogram pixels)")
    elif args.compute_stats:
        print("Not enough values to compute stats.")
        
if __name__ == "__main__":
    main()