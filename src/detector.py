"""One SHARED small autoencoder, applied to ONE channel at a time (channel-independent).

Window of W consecutive normalised values (one channel) -> encoder -> small code -> decoder -> W values.
Anomaly score at time t = mean squared reconstruction error of the window ENDING at t (uses only the past).
"""
import numpy as np
import torch
import torch.nn as nn

W = 48          # window length = 48 * 5 min = 4 h  (our choice)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class WindowAE(nn.Module):
    def __init__(self, w=W, hidden=64, code=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(w, hidden), nn.ReLU(), nn.Linear(hidden, code))
        self.dec = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, w))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_model(Z_train: np.ndarray, steps=3000, batch=2048, lr=1e-3, seed=0, log=print) -> WindowAE:
    """Z_train: [T, n_channels] normalised TRAIN period (no labels). Each step: random (channel, start) windows."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Zt = torch.from_numpy(Z_train).to(DEVICE)                 # [T, C]
    T, C = Zt.shape
    model = WindowAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ar = torch.arange(W, device=DEVICE)
    for s in range(steps):
        ch = torch.randint(0, C, (batch,), generator=g).to(DEVICE)
        st = torch.randint(0, T - W, (batch,), generator=g).to(DEVICE)
        x = Zt[st[:, None] + ar[None, :], ch[:, None]]        # [batch, W]
        loss = ((model(x) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 500 == 0 or s == steps - 1:
            log(f"  step {s:5d}  loss {loss.item():.4f}")
    return model.eval()


@torch.no_grad()
def score_channel(model: WindowAE, z: np.ndarray, chunk=200_000) -> np.ndarray:
    """z: [T] normalised channel -> scores [T] (first W-1 entries 0)."""
    zt = torch.from_numpy(z).to(DEVICE)
    out = torch.zeros(len(z), device=DEVICE)
    wins = zt.unfold(0, W, 1)                                  # [T-W+1, W] (a view)
    for a in range(0, wins.shape[0], chunk):
        x = wins[a:a + chunk]
        out[W - 1 + a: W - 1 + a + len(x)] = ((model(x) - x) ** 2).mean(1)
    return out.cpu().numpy()


@torch.no_grad()
def baseline_score_channel(z: np.ndarray) -> np.ndarray:
    """Sanity baseline (no learning): max |z| over the same window."""
    zt = torch.from_numpy(np.abs(z)).to(DEVICE)
    out = torch.zeros(len(z), device=DEVICE)
    out[W - 1:] = zt.unfold(0, W, 1).max(1).values
    return out.cpu().numpy()
