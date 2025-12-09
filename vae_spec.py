'''
    VAE model and training code for mel-spectrograms in .npy format.
'''
import os, math, random, argparse, glob, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

class NPYDataset(Dataset):
    '''
    Docstring for NPYDataset
    Loads .npy spectrogram files, normalizes them to [0,1] from [-80,0] dB,
    applies random cropping to target frames for sampling, and randomly masks
    time and frequency bands during training for data augmentation
    '''
    def __init__(self, root, target_frames, time_mask = 0, freq_mask = 0, split_ratio =(0.8, 0.1, 0.1), split = 'train', seed = 42, recursive = True):
        self.root = Path(root)
        self.target_frames = target_frames
        pat = "**/*.npy" if recursive else "*.npy"
        files = sorted([Path(p) for p in glob.glob(str(self.root / pat), recursive = recursive)])
        rng = random.Random(seed)
        idx = list(range(len(files)))
        rng.shuffle(idx)
        n = len(files)
        n_train = int(n*split_ratio[0])
        n_val = int(n*split_ratio[1])
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]
        if split == 'train':
            self.files = [files[i] for i in train_idx]
        elif split == 'val':
            self.files = [files[i] for i in val_idx]
        else:
            self.files = [files[i] for i in test_idx]
        self.time_mask = time_mask
        self.freq_mask = freq_mask
    
    def __len__(self):
        return(len(self.files))
    
    def _random_crop(self, x):
        H, W = x.shape
        T = self.target_frames
        if W == T:
            return x
        if W > T:
            start = np.random.randint(0, W - T + 1)
            return x[:, start:start + T]
        out = np.zeros((H, T), dtype = x.dtype)
        out[:, :W] = x
        return out
    
    def _specaugment(self, x):
        H, W = x.shape
        for _ in range(self.time_mask):
            w = np.random.randint(1, max(2, W // 10))
            t0 = np.random.randint(0, max(1, W- w + 1))
            x[:, t0:t0+w] = 0.0
        for _ in range(self.freq_mask):
            h = np.random.randint(1, max(2, H // 10))
            f0 = np.random.randint(0, max(1, H - h + 1))
            x[f0:f0+h, :] = 0.0
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

    
class Encoder(nn.Module):
    """
    Docstring for Encoder
    VAE Encoder for mel-spectrograms:
    Maps input mel-spectrograms ([B,1,128,172]) to latent mean and log-variance (mu, logvar)
    Uses 3 2d convolutional layers with stride = 2, GroupNorm, and SiLU activation
    Inferences on dummy input to compute flattened feature dimension and create
    fc_mu(mean of latent space) and fc_logvar (log-variance)
    """
    def __init__(self, in_ch = 1, h = 128, w = 172, z_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  kernel_size=3, stride=2, padding=1), nn.GroupNorm(4,32), nn.SiLU(),
            nn.Conv2d(32,  64,    kernel_size=3, stride=2, padding=1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.Conv2d(64,  128,   kernel_size=3, stride=2, padding=1), nn.GroupNorm(16,128), nn.SiLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, h, w)
            out = self.net(dummy)
            self.feat_shape = out.shape[1:]
            self.flat_dim = int(np.prod(self.feat_shape))
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x):
        f = self.net(x)
        f = f.flatten(1)
        mu = self.fc_mu(f)
        logvar = self.fc_logvar(f)
        return mu, logvar

class Decoder(nn.Module):
    """
    Docstring for Decoder
    Takes latent vector z and reconstructs an output spectrogram
    Uses a linear layer to expand latent vector back to the feature map size
    Uses 3 ConvTranspose2d layers with stride = 2, GroupNorm, and SiLU activation
    The final Conv2d layer maps to an ouput channel with Sigmoid activation to scale to [0,1]
    """
    def __init__(self, out_ch=1, h=128, w=172, z_dim=128, feat_shape=(128,16,22)):
        super().__init__()
        C, Hf, Wf = feat_shape
        self.fc = nn.Linear(z_dim, C*Hf*Wf)
        self.Hf, self.Wf, self.C = Hf, Wf, C
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C, 64,  kernel_size=4, stride=2, padding=1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.ConvTranspose2d(64,32,  kernel_size=4, stride=2, padding=1), nn.GroupNorm(4,32), nn.SiLU(),
            nn.ConvTranspose2d(32,16,  kernel_size=4, stride=2, padding=1), nn.GroupNorm(4,16), nn.SiLU(),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.target_h = h
        self.target_w = w
    def forward(self, z):
        f = self.fc(z).view(z.size(0), self.C, self.Hf, self.Wf)
        x = self.net(f)
        x = x[:, :, :self.target_h, :self.target_w]
        return x


class VAE(nn.Module):
    """
    Docstring for VAE
    Calls Encoder and Decoder to create a full VAE model
    Implements the reparameterization trick in parameterize() and forward pass in forward()
    """
    def __init__(self, h = 128, w = 172, z_dim = 128):
        super().__init__()
        self.encode = Encoder(in_ch = 1, h = h, w = w, z_dim = z_dim)
        self.decode = Decoder(out_ch = 1, h = h, w = w, z_dim = z_dim, feat_shape = self.encode.feat_shape)
    def parameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x)
        x = self.parameterize(mu, logvar)
        x_hat = self.decode(x)
        return x_hat, mu, logvar

def vae_loss(recon, x, mu, logvar, beta = 1.0, recon_type = 'l1'):
    """
    Docstring for vae_loss
    Returns the VAE loss as sum of reconstruction loss and KL divergence, factoring in a parameter beta
    """
    if recon_type == 'bce':
        recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction = 'mean')
    elif recon_type == 'mse':
        recon_loss = nn.functional.mse_loss(recon, x, reduction = 'mean')
    else:
        recon_loss = nn.functional.l1_loss(recon, x, reduction = 'mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

"""
save_recon: saves original and reconstructed spectrograms from the model on a validation set
save_prior: samples random latent z and decodes to generate synthetic mel-spectrograms
"""

def save_recon(model, loader, device, out_dir, n_batches = 1):
    out_dir.mkdir(parents = True, exist_ok = True)
    model.eval()
    taken = 0
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

def save_prior(model, device, out_dir, n = 16):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    z_dim = model.encode.fc_mu.out_features
    z = torch.randn(n, z_dim, device=device)
    x = model.decode(z)
    for i in range(n):
        np.save((out_dir / f"sample_{i:04d}.npy").as_posix(),
                x[i].squeeze(0).detach().cpu().numpy().astype(np.float16))
        
def main():
    """
    Docstring for main
    Argument parsing: takes arguments from terminal input to set up parameters
    Create and load dataset, dataloader, model, optimizer, and training loop
    Per-epoch metrics saved to metrics.csv
    Saves reconstructions, prior samples every 3 epochs
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing .npy spectrograms")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="Max β for KL (after warmup)")
    ap.add_argument("--kl_warmup_epochs", type=int, default=5)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--target_frames", type=int, default=172)
    ap.add_argument("--specaug_time_mask", type=int, default=0)
    ap.add_argument("--specaug_freq_mask", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--recon", type=str, default="l1", choices=["l1","mse","bce"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok = True)
    train_ds = NPYDataset(args.data_dir, target_frames = args.target_frames, time_mask = args.specaug_time_mask, freq_mask = args.specaug_freq_mask, split = 'train')
    val_ds = NPYDataset(args.data_dir, target_frames = args.target_frames, split = 'val')
    train_load = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
    val_load = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory=True)
    metrics_csv = (save_dir / "metrics.csv").as_posix()
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","beta","train_loss","train_recon","train_kl","val_loss","val_recon","val_kl"])
    model = VAE(h=args.n_mels, w=args.target_frames, z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val = float('inf')
    global_step = 0
    # Training loop
    for epoch in range(1, args.epochs + 1):
        """
        Trains the VAE model for a specified number of epochs
        Implements mixed precision training using torch.cuda.amp for efficiency
        KL warmup is applied to gradually increase the weight of beta over the first few epochs
        After each epoch, evaluates on validation set and saves best model checkpoint
        """
        beta_now = args.beta * min(1.0, epoch/max(1, args.kl_warmup_epochs))
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        t0 = time.time()
        for x, _ in train_load:
            x = x.to(device, non_blocking = True)
            opt.zero_grad(set_to_none = True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                x_hat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=beta_now, recon_type = args.recon)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running['loss'] += loss.item()
            running['recon'] += recon.item()
            running['kl'] += kl.item()
            global_step += 1
        n_batches = max(1, len(train_load))
        tr_loss = running['loss']/n_batches
        tr_recon = running['recon']/n_batches
        tr_kl = running['kl']/n_batches

        model.eval()
        with torch.no_grad():
            vr = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            for x, _ in val_load:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                vloss, vrecon, vkl = vae_loss(x_hat, x, mu, logvar, beta=beta_now, recon_type = args.recon)
                vr['loss'] += vloss.item()
                vr['recon'] += vrecon.item()
                vr['kl'] += vkl.item()
            nvb = max(1, len(val_load))
            val_loss = vr['loss']/nvb
            val_recon = vr['recon']/nvb
            val_kl = vr['kl']/nvb

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | beta {beta_now:.3f} | "
              f"train: loss {tr_loss:.4f} rec {tr_recon:.4f} kl {tr_kl:.4f} | "
              f"val: loss {val_loss:.4f} rec {val_recon:.4f} kl {val_kl:.4f} | {dt:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "beta_now": beta_now
            }
            torch.save(ckpt, (save_dir / "vae_mel_best.pt").as_posix())
        if epoch % 3 == 0:
            save_recon(model, val_load, device, save_dir / f"samples_epoch{epoch}")
            save_prior(model, device, save_dir / f"samples_epoch{epoch}", n=16)
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, beta_now, tr_loss, tr_recon, tr_kl, val_loss, val_recon, val_kl])

    print("Done. Best val loss:", best_val)
    print("Best checkpoint:", save_dir / "vae_mel_best.pt")



if __name__ == "__main__":
    main()
