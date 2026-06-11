# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

import logging
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tversky import nn as tnn

_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

_OUT   = pathlib.Path(__file__).parent / "outputs" / "mnist"
_log   = logging.getLogger(__name__)


class MnistNet(nn.Module):
    def __init__(self, fbank_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding="same")
        self.conv2 = nn.Conv2d(12, 12, 3, padding="same")
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(12, 12, 3, padding="same")
        self.conv4 = nn.Conv2d(12, 12, 3, padding="same")
        self.pool4 = nn.MaxPool2d(2)


        self.conv5 = nn.Conv2d(12, 12, 3, padding="same")
        self.conv6 = nn.Conv2d(12, 36, 3, padding="same")

        self.tproj  = tnn.TverskyProjection(
            embedding_dim=36,
            class_count=10,
            fbank_size=fbank_size,
            similarity_model="contrast",
            intersection_reduction="product", 
            difference_reduction="ignorematch",
            normalize=False,
        )

    def forward_conv(self, x):
        x = self.conv2(F.relu(self.conv1(x)))
        # h2 = x.mean((-2, -1))
        x = self.pool2(x)

        x = self.conv4(F.relu(self.conv3(x)))
        # h4 = x.mean((-2, -1))
        x = self.pool4(x)

        x = self.conv6(F.relu(self.conv5(x)))
        x = x.mean((-2, -1))

        #x = torch.concatenate([h4, h6], dim=-1)

        return x

    def forward(self, x):
        x = self.forward_conv(x)
        return self.tproj(x)
    
    def compute_salience(self, x):
        x = self.forward_conv(x)
        # (b, d) @ (f, d).T = (b, f)
        feature_measures = x @ self.tproj.feature_bank.weight.T
        salience_measures = F.relu(feature_measures).sum(-1)
        return salience_measures



def _plot_training(loss_curve, gnorm_curve, acc_curve, alpha_curve, beta_curve, theta_curve):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    ax = axes[0]
    ax.set_title("Loss & Grad Norm over Updates")
    ax.plot(loss_curve, color="#2266CC", linewidth=0.8, label="loss")
    ax.set_xlabel("update")
    ax.set_ylabel("loss", color="#2266CC")
    ax.tick_params(axis="y", colors="#2266CC")
    ax2 = ax.twinx()
    ax2.plot(gnorm_curve, color="#CC6600", linewidth=0.8, alpha=0.8, label="grad norm")
    ax2.set_ylabel("grad norm", color="#CC6600")
    ax2.tick_params(axis="y", colors="#CC6600")

    axes[1].set_title("Val Accuracy over Epochs")
    axes[1].plot(range(1, len(acc_curve) + 1), [a * 100 for a in acc_curve],
                 marker="o", color="#227722", linewidth=1.2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(95, color="#CC0000", linestyle="--", linewidth=0.8, label="95% threshold")
    axes[1].legend(fontsize=8)

    axes[2].set_title("Tversky α, β, θ over Updates")
    axes[2].plot(alpha_curve, color="#AA2299", linewidth=0.9, label="α")
    axes[2].plot(beta_curve,  color="#2299AA", linewidth=0.9, label="β")
    axes[2].plot(theta_curve, color="#999922", linewidth=0.9, label="θ")
    axes[2].set_xlabel("update")
    axes[2].set_ylabel("parameter value")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = _OUT / "training.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_salience(model, val_loader):
    n_each = 5   # low / mid / high
    n_cols = n_each * 3

    records = []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(_DEVICE), y.to(_DEVICE)
            s = model.compute_salience(x).cpu()
            for sal, img, lbl in zip(s, x.cpu(), y.cpu()):
                records.append((sal.item(), img, lbl.item()))

    by_class = {c: [] for c in range(10)}
    for sal, img, lbl in records:
        by_class[lbl].append((sal, img))
    for c in by_class:
        by_class[c].sort(key=lambda t: t[0])

    fig, axes = plt.subplots(10, n_cols, figsize=(n_cols * 1.1, 10 * 1.1))
    fig.suptitle("Salience: lowest 5 | mid 5 | highest 5  (1 row per class)",
                 fontsize=10, fontweight="bold")

    for row, cls in enumerate(range(10)):
        items = by_class[cls]
        n = len(items)
        mid_start = (n - n_each) // 2
        indices = (list(range(n_each))
                   + list(range(mid_start, mid_start + n_each))
                   + list(range(n - n_each, n)))
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(str(cls), fontsize=8, rotation=0, labelpad=12, va="center")
            sal, img = items[idx]
            ax.imshow(img.squeeze(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"{sal:.1f}", fontsize=5, pad=1)

    fig.tight_layout()
    # Two vertical separators between low|mid and mid|high
    for sep_col in (n_each, n_each * 2):
        x_left  = axes[0, sep_col - 1].get_position().x1
        x_right = axes[0, sep_col].get_position().x0
        x_mid   = (x_left + x_right) / 2
        fig.add_artist(plt.Line2D(
            [x_mid, x_mid], [0.02, 0.96],
            transform=fig.transFigure, color="#444444", linewidth=2.0, linestyle="--"
        ))
    path = _OUT / "salience.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_misclassifications(model, val_loader):
    n_per_class = 6
    # misses[true_class] = list of (true_logit, probs_cpu, pred_class)
    misses = {c: [] for c in range(10)}

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(_DEVICE), y.to(_DEVICE)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            wrong  = preds != y
            for img, true, pred, lg, pr in zip(x[wrong].cpu(), y[wrong].cpu(),
                                               preds[wrong].cpu(), logits[wrong].cpu(),
                                               probs[wrong].cpu()):
                true_c = true.item()
                misses[true_c].append((lg[true_c].item(), img, pr, pred.item()))

    for c in misses:
        misses[c].sort(key=lambda t: t[0])  # ascending: lowest correct-class logit first

    classes = list(range(10))
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(n_per_class * 1.6, 10 * 2.2))
    fig.suptitle("Top misclassifications — output class distribution (true=green, pred=red)",
                 fontsize=10, fontweight="bold")
    # Outer grid: one row per class, with extra hspace between rows for dividers
    outer_gs = GridSpec(10, 1, figure=fig, hspace=0.35)

    for row, true_class in enumerate(range(10)):
        inner_gs = GridSpecFromSubplotSpec(
            2, n_per_class,
            subplot_spec=outer_gs[row],
            height_ratios=[1, 3], hspace=0.08, wspace=0.4,
        )
        for col in range(n_per_class):
            bar_ax = fig.add_subplot(inner_gs[0, col])
            img_ax = fig.add_subplot(inner_gs[1, col])
            bar_ax.axis("off")
            img_ax.axis("off")
            if col == 0:
                img_ax.set_ylabel(f"true={true_class}", fontsize=7, rotation=0,
                                  labelpad=32, va="center")
            if col < len(misses[true_class]):
                true_logit, img, probs, pred = misses[true_class][col]

                img_ax.imshow(img.squeeze(), cmap="gray_r", vmin=0, vmax=1)

                colors = ["#CCCCCC"] * 10
                colors[true_class] = "#22AA44"
                colors[pred]       = "#CC2222"
                bar_ax.axis("on")
                bar_ax.bar(classes, probs.tolist(), color=colors, width=0.8, linewidth=0)
                bar_ax.set_xlim(-0.5, 9.5)
                bar_ax.set_ylim(0, 1)
                bar_ax.set_xticks(classes)
                bar_ax.tick_params(labelsize=4, length=1, pad=0)
                bar_ax.set_yticks([])
                for spine in bar_ax.spines.values():
                    spine.set_linewidth(0.3)
                bar_ax.set_title(f"pred={pred} ({true_logit:.2f})", fontsize=6, pad=2)

        # Horizontal divider below this class row (skip after the last row)
        if row < 9:
            spec = outer_gs[row]
            y_bottom = spec.get_position(fig).y0
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [y_bottom, y_bottom],
                transform=fig.transFigure, color="#AAAAAA", linewidth=0.8, linestyle="-",
            ))

    path = _OUT / "misclassifications.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(model, acc, last_loss, training_path, salience_path, misclass_path):
    n_params = sum(p.numel() for p in model.parameters())
    model_repr = repr(model)
    report = f"""\
# MNIST Test Report

## Model

| | |
|---|---|
| Parameters | {n_params:,} |
| Device | {_DEVICE} |

```
{model_repr}
```

## Results

| | |
|---|---|
| Val accuracy | {acc:.2%} |
| Last train loss | {last_loss:.4f} |
| Threshold | 95.00% |
| Pass | {"✓" if acc >= 0.95 else "✗"} |

## Training curves

![Loss, grad norm, and val accuracy]({training_path.name})

## Salience

One row per class (0–9): 5 lowest salience | 5 mid salience | 5 highest salience. Salience value shown above each image.

![Salience]({salience_path.name})

## Misclassifications

One row per true class (0–9). Each cell shows an image the model got wrong, with the predicted label in red.

![Misclassifications]({misclass_path.name})
"""
    path = _OUT / "report.md"
    path.write_text(report)
    return path


def test_mnist():
    torch.manual_seed(0)
    _OUT.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(_OUT / "test_mnist.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("/tmp/mnist", train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)

    model    = MnistNet(fbank_size=36).to(_DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    _log.info("device: %s", _DEVICE)
    _log.info("model parameters: %d", n_params)

    n_epochs    = 1000
    opt         = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched       = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_curve  = []
    gnorm_curve = []
    acc_curve   = []
    alpha_curve = []
    beta_curve  = []
    theta_curve = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(_DEVICE), y.to(_DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            gnorm = sum(p.grad.norm() ** 2 for p in model.parameters()
                        if p.grad is not None) ** 0.5
            opt.step()
            loss_curve.append(loss.item())
            gnorm_curve.append(gnorm.item())
            alpha_curve.append(model.tproj.alpha.item())
            beta_curve.append(model.tproj.beta.item())
            theta_curve.append(model.tproj.theta.item())
            epoch_losses.append(loss.item())

        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(_DEVICE), y.to(_DEVICE)
                logits = model(x)
                val_loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        epoch_acc      = correct / total
        train_loss_mean = sum(epoch_losses) / len(epoch_losses)
        val_loss_mean   = val_loss_sum / total
        acc_curve.append(epoch_acc)
        sched.step()
        _log.info("epoch %2d  train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%  lr=%.2e",
                  epoch + 1, train_loss_mean, val_loss_mean, epoch_acc * 100,
                  sched.get_last_lr()[0])

    acc        = acc_curve[-1]
    last_loss  = loss_curve[-1]
    _log.info("final val accuracy: %.2f%%  last train loss: %.4f", acc * 100, last_loss)

    training_path = _plot_training(loss_curve, gnorm_curve, acc_curve,
                                   alpha_curve, beta_curve, theta_curve)
    salience_path = _plot_salience(model, val_loader)
    misclass_path = _plot_misclassifications(model, val_loader)
    report_path   = _write_report(model, acc, last_loss, training_path, salience_path, misclass_path)
    _log.info("outputs written to %s", _OUT)

    assert acc >= 0.95, f"MNIST val accuracy {acc:.2%} < 95%"
