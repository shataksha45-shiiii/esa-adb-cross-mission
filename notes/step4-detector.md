# Step 4: First detector (one shared small autoencoder) and the scorer

## What we did
1. Installed PyTorch 2.15 (and `portion`, needed by the official metric). Confirmed the Apple GPU backend (`mps`) is available.
2. Wrote `src/detector.py`:
   - `WindowAE`: a small autoencoder.
   - `train_model`: trains it on the training period only, no labels.
   - `score_channel`: turns a channel into an anomaly score at every time step.
   - `baseline_score_channel`: a no-learning sanity baseline.

## Why
Label-free detection: learn what *normal-looking* windows look like, then flag windows the model cannot reproduce. One **shared** model for all channels (pooled), applied **one channel at a time**, so it can be moved from one mission to another.

## Key terms
- **Window**: W consecutive values of one channel. We use W = 48 steps = 4 hours (our choice).
- **Autoencoder**: a network that squeezes the window into a few numbers (the **code**) and tries to rebuild it. If it learned normal patterns, normal windows rebuild well and strange windows rebuild badly.
- **Reconstruction error**: how far the rebuilt window is from the real one.
- **Anomaly score** at time t: error of the window that **ends** at t (it only uses the past).
- **PyTorch / tensor / mps**: PyTorch is the deep learning library, a tensor is its n-dimensional array, `mps` runs it on the Mac GPU.
- **Epoch / step / batch**: one step = one update using one batch of examples.

## The maths

For a window x of W values and the rebuilt window x-hat:

$$
\text{loss} = \frac{1}{W}\sum_{k=1}^{W}\big(x_k - \hat{x}_k\big)^2,
\qquad
s_t = \frac{1}{W}\sum_{k=0}^{W-1}\big(z_{t-k} - \hat{z}_{t-k}\big)^2
$$

The training loss is the average of this over a batch. The score s_t is the same quantity for the window ending at time t.

## Model and training (our choices, not tuned)
- Network: 48 -> 64 -> 8 -> 64 -> 48 (ReLU in between). About 7k parameters.
- Training data: the **training period** of one mission, all its target channels pooled. Each step samples 2048 random (channel, start time) windows.
- 6000 steps, Adam, learning rate 1e-3, 3 random seeds (0, 1, 2).
- **Training data contains unlabelled anomalies** (the benchmark training sets are not clean). The model sees them as ordinary data. That is the label-free setting.

## Code, line by line (core)
```python
x = Zt[st[:, None] + ar[None, :], ch[:, None]]   # gather windows: row = start+0..47, column = channel
loss = ((model(x) - x) ** 2).mean()               # reconstruction error
opt.zero_grad(); loss.backward(); opt.step()      # backprop: adjust weights to lower the loss
```
```python
wins = zt.unfold(0, W, 1)                         # all sliding windows as a view, no copy
out[W-1+a : ...] = ((model(x) - x) ** 2).mean(1)  # score per window, written at the window's END time
```
- `unfold(0, W, 1)` slides a window of size W with stride 1.
- First W-1 scores are 0 because no full window exists yet; we ignore them in calibration.

## Baseline (no learning)
`max |z|` over the same window. If the autoencoder cannot beat this, the network adds nothing.

## What broke
Nothing at the code level. Honest result note: see Step 6. The detector is basic and untuned.

## Flags
- W, network size, steps, learning rate: chosen once, **not tuned**.
- No train/validation split inside the training period was used to stop early (the paper holds out the last 3 months of train as validation; we did not use it).
- A single-channel model cannot see relationships between channels (a failure only visible across several channels will be missed).

## Self-test
1. Why does the score use the window that *ends* at t instead of one centred on t?
2. What happens to a normal autoencoder if a lot of anomalies sit in its training data?
3. Why is `max |z|` a fair baseline to compare against?
