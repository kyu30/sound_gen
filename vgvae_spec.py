import os, math, random, argparse, glob, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv

class NPYDataset(Dataset):
    def __init__(self, root, target_frames, time_mask=0, freq_mask=0,
                 split_ratio=(0.8, 0.1, 0.1), split='train', seed=42, recursive=True):
        self.root = Path(root)
        self.target_frames = target_frames
        pat = "**/*.npy" if recursive else "*.npy"
        files = sorted([Path(p) for p in glob.glob(str(self.root / pat), recursive=recursive)])
        rng = random.Random(seed)
        idx = list(range(len(files)))
        rng.shuffle(idx)
        n = len(files)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        if split == 'train':
            self.files = [files[i] for i in train_idx]
        elif split == 'val':
            self.files = [files[i] for i in val_idx]
        else:
            self.files = [files[i] for i in test_idx]
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def __len__(self):
        return len(self.files)

    def _random_crop(self, x: np.ndarray) -> np.ndarray:
        H, W = x.shape
        T = self.target_frames
        if W == T:
            return x
        if W > T:
            start = np.random.randint(0, W - T + 1)
            return x[:, start:start + T]
        out = np.zeros((H, T), dtype=x.dtype)
        out[:, :W] = x
        return out

    def _specaugment(self, x: np.ndarray) -> np.ndarray:
        H, W = x.shape
        for _ in range(self.time_mask):
            w = np.random.randint(1, max(2, W // 10))
            t0 = np.random.randint(0, max(1, W - w + 1))
            x[:, t0:t0 + w] = 0.0
        for _ in range(self.freq_mask):
            h = np.random.randint(1, max(2, H // 10))
            f0 = np.random.randint(0, max(1, H - h + 1))
            x[f0:f0 + h, :] = 0.0
        return x

    def __getitem__(self, idx):
        path = self.files[idx]
        mel = np.load(path.as_posix()).astype(np.float32)
        mel = np.clip(mel, -80.0, 0.0)
        mel = (mel + 80.0) / 80.0
        mel = self._random_crop(mel)
        if "train" in str(self.root).lower() or "training" in str(self.root).lower():
            if self.time_mask or self.freq_mask:
                mel = self._specaugment(mel)
        return torch.from_numpy(mel)[None, :, :], path.name

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=512, code_dim=128, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.embeddings = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)

        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embeddings.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        codes = torch.argmin(dist, dim=1)
        z_q = self.embeddings(codes).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()
        codes = codes.view(B, H, W)
        return z_q_st, vq_loss, codes

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1),  # (1,128,172)->(32,64,86)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),     # ->(64,32,43)
            nn.ReLU(),
            nn.Conv2d(64, z_ch, 4, stride=2, padding=1),   # ->(z_ch,16,22)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch, 64, 4, stride=2, padding=1),   # (z_ch,16,22)->(64,32,44)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # ->(32,64,88)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),     # ->(16,128,176)
            nn.ReLU(),
            nn.Conv2d(16, out_ch, 3, padding=1),                    # ->(1,128,176)
            nn.Sigmoid(),
        )
    def forward(self, z_q):
        x_hat = self.net(z_q)
        x_hat = x_hat[:, :, :, :172]
        return x_hat
    
class VQVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_ch=128, num_codes=512, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, z_ch=z_ch)
        self.quantizer = VectorQuantizer(num_codes=num_codes, code_dim=z_ch, beta=beta)
        self.decoder = Decoder(out_ch=out_ch, z_ch=z_ch)
        self.num_codes = num_codes
        self.z_ch = z_ch

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, codes = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, codes


def vqvae_loss(x, x_hat, vq_loss, recon_type="l1"):
    T = min(x.shape[-1], x_hat.shape[-1])
    x = x[..., :T]
    x_hat = x_hat[..., :T]
    if recon_type == "l1":
        recon = F.l1_loss(x_hat, x)
    elif recon_type == "mse":
        recon = F.mse_loss(x_hat, x)
    else:
        recon = F.binary_cross_entropy(x_hat, x)
    return recon + vq_loss, recon, vq_loss

def pad(batch):
    xs, names = zip(*batch)
    x = torch.stack(xs, dim=0)
    return x, names


def save_recon(model, loader, device, out_dir, n_batches=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    taken = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, _, _ = model(x)
            for i in range(min(x.size(0), 8)):
                np.save((out_dir / f"recon_{taken:04d}_orig.npy").as_posix(),
                        x[i].squeeze(0).cpu().numpy().astype(np.float16))
                np.save((out_dir / f"recon_{taken:04d}_reco.npy").as_posix(),
                        x_hat[i].squeeze(0).detach().cpu().numpy().astype(np.float16))
                taken += 1
            n_batches -= 1
            if n_batches <= 0:
                break


def save_prior(model, device, out_dir, n=16):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 128, 172, device=device)
        z_e = model.encoder(dummy)
        _, z_ch, H, W = z_e.shape
        codes = torch.randint(0, model.num_codes, (n, H, W), device=device)  # (B,H,W)
        z_q = model.quantizer.embeddings(codes.view(-1)) \
                              .view(n, H, W, z_ch) \
                              .permute(0, 3, 1, 2).contiguous()
        x = model.decoder(z_q)
        for i in range(n):
            np.save((out_dir / f"sample_{i:04d}.npy").as_posix(),
                    x[i].squeeze(0).detach().cpu().numpy().astype(np.float16))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing .npy spectrograms")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--target_frames", type=int, default=172)
    ap.add_argument("--specaug_time_mask", type=int, default=0)
    ap.add_argument("--specaug_freq_mask", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--recon", type=str, default="l1", choices=["l1", "mse", "bce"])
    ap.add_argument("--num_codes", type=int, default=512)
    ap.add_argument("--vq_beta", type=float, default=0.25)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    train_ds = NPYDataset(args.data_dir, target_frames=args.target_frames,
                          time_mask=args.specaug_time_mask,
                          freq_mask=args.specaug_freq_mask, split='train')
    val_ds = NPYDataset(args.data_dir, target_frames=args.target_frames,
                        split='val')
    train_load = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=pad, pin_memory=True)
    val_load = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=pad, pin_memory=True)
    metrics_csv = (save_dir / "metrics.csv").as_posix()
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_recon", "train_vq",
                    "val_loss", "val_recon", "val_vq"])
    model = VQVAE(in_ch=1, out_ch=1, z_ch=args.z_dim,
                  num_codes=args.num_codes, beta=args.vq_beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "vq": 0.0}
        t0 = time.time()
        for x, _ in train_load:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                x_hat, vq_loss, _ = model(x)
                loss, recon, vq = vqvae_loss(x, x_hat, vq_loss, recon_type=args.recon)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running['loss'] += loss.item()
            running['recon'] += recon.item()
            running['vq'] += vq.item()
        n_batches = max(1, len(train_load))
        tr_loss = running['loss'] / n_batches
        tr_recon = running['recon'] / n_batches
        tr_vq = running['vq'] / n_batches
        model.eval()
        vr = {"loss": 0.0, "recon": 0.0, "vq": 0.0}
        with torch.no_grad():
            for x, _ in val_load:
                x = x.to(device)
                x_hat, vq_loss, _ = model(x)
                vloss, vrecon, vvq = vqvae_loss(x, x_hat, vq_loss, recon_type=args.recon)
                vr['loss'] += vloss.item()
                vr['recon'] += vrecon.item()
                vr['vq'] += vvq.item()
        nvb = max(1, len(val_load))
        val_loss = vr['loss'] / nvb
        val_recon = vr['recon'] / nvb
        val_vq = vr['vq'] / nvb
        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | "
              f"train: loss {tr_loss:.4f} rec {tr_recon:.4f} vq {tr_vq:.4f} | "
              f"val: loss {val_loss:.4f} rec {val_recon:.4f} vq {val_vq:.4f} | {dt:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, (save_dir / "vqvae_mel_best.pt").as_posix())
        if epoch % 3 == 0:
            save_recon(model, val_load, device, save_dir / f"samples_epoch{epoch}")
            save_prior(model, device, save_dir / f"samples_epoch{epoch}", n=16)
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_loss, tr_recon, tr_vq,
                        val_loss, val_recon, val_vq])
    print("Done. Best val loss:", best_val)
    print("Best checkpoint:", save_dir / "vqvae_mel_best.pt")


if __name__ == "__main__":
    main()
