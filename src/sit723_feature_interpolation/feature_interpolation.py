"""SIT723 baseline: feature-interpolation clean-label poisoning, inverted to pixel space via PGD.

Pipeline (paper Section III-B):
  1. Surrogate ResNet-18 fine-tuned on TARGET-class images only; prototype p_t = mean penultimate embedding.
  2. For a base-class image x_b: z_b = embed(x_b);  z~ = (1 - lam) * z_b + lam * p_t   (Eq. 1)
  3. PGD (L-inf budget eps around x_b) finds x~_b with embed(x~_b) ~= z~.  Label of x~_b stays the base label.
  4. Victim ResNet-18 is fine-tuned on clean + poisoned data; CDA / ASR / SSIM are measured.

NOTE (naming): in this phase target_class = cat (3) and base_class = dog (5) by default, i.e. the opposite
role assignment to Phase 2 (where poisons are cats and the attacked images are dogs).
NOTE: both surrogate and victim start from ImageNet-pretrained ResNet-18 weights (pretrained=True).
NOTE: no LPIPS term is used here - only the L2 embedding loss and the L-inf projection.
"""
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18
from tqdm import tqdm

from shared_utils.data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, class_indices

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. SSIM disabled.")


# ── normalisation helpers ────────────────────────────────────────────────────
def denormalize(tensor):
    """Normalised -> [0,1] pixel space (used for PGD clipping / visualisation)."""
    mean = torch.tensor(CIFAR10_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def renormalize(tensor):
    mean = torch.tensor(CIFAR10_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


# ── data wrapper ─────────────────────────────────────────────────────────────
class PoisonedDataset(Dataset):
    """Replaces a subset of base-class images by their poisoned counterparts, labels unchanged."""

    def __init__(self, base_dataset, base_class_indices, poisoned_images, poison_indices):
        self.dataset = base_dataset
        self.base_idx = base_class_indices
        self.poisoned_images = poisoned_images      # dict: train idx -> poisoned tensor
        self.poison_indices = set(poison_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.poison_indices:
            img = self.poisoned_images[idx]
        return img, label


# ── model ────────────────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    """ResNet-18 with a new FC head; .embed(x) returns the 512-d penultimate feature."""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # -> (B, 512, 1, 1)
        self.fc = nn.Linear(512, num_classes)

    def embed(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, x):
        return self.fc(self.embed(x))


# ── Step 1: surrogate + prototype ────────────────────────────────────────────
def train_surrogate(model, train_dataset, target_class, epochs, batch_size, device):
    """Fine-tune the surrogate on target-class images only."""
    print(f"\n[Phase 1] Fine-tuning surrogate on class {target_class} only...")
    subset = Subset(train_dataset, class_indices(train_dataset, target_class))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"  Surrogate epoch {ep + 1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {ep + 1:2d}  loss={total_loss / len(loader):.4f}")
    return model


def extract_prototype(model, train_dataset, target_class, batch_size, device):
    """p_t = mean embedding of all target-class training images."""
    print("\n[Phase 1] Extracting target-class prototype...")
    subset = Subset(train_dataset, class_indices(train_dataset, target_class))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  Encoding target class", leave=False):
            embeddings.append(model.embed(imgs.to(device)).cpu())
    prototype = torch.cat(embeddings, dim=0).mean(0)  # (512,)
    print(f"  prototype shape: {prototype.shape}, norm: {prototype.norm():.3f}")
    return prototype


# ── Step 2: interpolation + PGD inversion ────────────────────────────────────
def interpolate_latent(z_b, p_t, lam):
    """z~ = (1 - lam) z_b + lam p_t"""
    return (1 - lam) * z_b + lam * p_t


def feature_inversion_pgd(model, x_b, z_target, epsilon, pgd_steps, pgd_lr, device):
    """argmin ||f(x') - z_target||_2  s.t. ||x' - x_b||_inf <= eps  (optimised in [0,1] pixel space)."""
    model.eval()
    x_b, z_target = x_b.to(device), z_target.to(device)
    x_orig_01 = denormalize(x_b).detach().clone()
    x_adv_01 = x_orig_01.clone().requires_grad_(True)

    for _ in range(pgd_steps):
        z_curr = model.embed(renormalize(x_adv_01.clamp(0, 1)))
        loss = torch.norm(z_curr - z_target, p=2)
        loss.backward()
        with torch.no_grad():
            grad = x_adv_01.grad
            gnorm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1) + 1e-8
            x_adv_01 -= pgd_lr * grad / gnorm                         # normalised-gradient step
            delta = (x_adv_01 - x_orig_01).clamp(-epsilon, epsilon)   # project to eps-ball
            x_adv_01.copy_((x_orig_01 + delta).clamp(0, 1))
        x_adv_01.grad = None
    return renormalize(x_adv_01.detach().clamp(0, 1))


def stealthiness_check(x_orig, x_poison, epsilon):
    """Returns (l_inf_ok, ssim_score). L-inf is the hard constraint; SSIM is reported."""
    o = np.clip(denormalize(x_orig.cpu()).squeeze().permute(1, 2, 0).numpy(), 0, 1)
    p = np.clip(denormalize(x_poison.cpu()).squeeze().permute(1, 2, 0).numpy(), 0, 1)
    l_inf_ok = np.abs(o - p).max() <= epsilon + 1e-4
    ssim_score = ssim(o, p, channel_axis=2, data_range=1.0) if HAS_SKIMAGE else None
    return l_inf_ok, ssim_score


def build_poisoned_dataset(surrogate, train_dataset, base_class, prototype,
                           lam, epsilon, poison_rate, pgd_steps, pgd_lr, device):
    """Poison a random `poison_rate` fraction of the base class. Returns (dataset, stats dict)."""
    print(f"\n[Phase 2] Building poisoned dataset (lam={lam}, eps={epsilon:.4f}, rate={poison_rate})...")
    base_idx = class_indices(train_dataset, base_class)
    n_poison = int(len(base_idx) * poison_rate)
    poison_idx = random.sample(base_idx, n_poison)
    print(f"  Base class samples : {len(base_idx)}\n  Poisoning {n_poison} of them ({poison_rate * 100:.0f}%)")

    poisoned_images, fail_count = {}, 0
    p_t = prototype.unsqueeze(0).to(device)
    surrogate.eval()
    for idx in tqdm(poison_idx, desc="  Inverting interpolated embeddings"):
        x_b = train_dataset[idx][0].unsqueeze(0).to(device)
        with torch.no_grad():
            z_b = surrogate.embed(x_b)
        x_poison = feature_inversion_pgd(surrogate, x_b, interpolate_latent(z_b, p_t, lam),
                                         epsilon, pgd_steps, pgd_lr, device)
        ok, _ = stealthiness_check(x_b, x_poison, epsilon)
        fail_count += int(not ok)
        poisoned_images[idx] = x_poison.squeeze(0).cpu()

    print(f"  Done. L-inf violations: {fail_count}/{n_poison}")
    mean_ssim = None
    if HAS_SKIMAGE:
        sample = random.sample(list(poisoned_images.items()), min(20, len(poisoned_images)))
        scores = [stealthiness_check(train_dataset[i][0].unsqueeze(0), x.unsqueeze(0), epsilon)[1] for i, x in sample]
        mean_ssim = float(np.mean(scores))
        print(f"  Mean SSIM (sample of 20): {mean_ssim:.4f}")

    stats = {"n_poison": n_poison, "linf_violations": fail_count, "mean_ssim_sample20": mean_ssim}
    return PoisonedDataset(train_dataset, base_idx, poisoned_images, poison_idx), stats


def visualize_poison_pairs(train_dataset, poisoned_dataset, out_dir, n=8):
    """Original / poisoned / |diff|x10 for n random poisons (paper Fig. 4)."""
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    sample = random.sample(list(poisoned_dataset.poison_indices), min(n, len(poisoned_dataset.poison_indices)))
    label = None
    for col, idx in enumerate(sample):
        x_orig, label = train_dataset[idx]
        x_pois = poisoned_dataset.poisoned_images[idx]
        o = denormalize(x_orig.unsqueeze(0)).squeeze().permute(1, 2, 0).numpy().clip(0, 1)
        p = denormalize(x_pois.unsqueeze(0)).squeeze().permute(1, 2, 0).numpy().clip(0, 1)
        axes[0, col].imshow(o); axes[1, col].imshow(p)
        axes[2, col].imshow((np.abs(o - p) * 10).clip(0, 1), cmap="inferno")
        for r in range(3):
            axes[r, col].axis("off")
    plt.suptitle(f"Class {label} ({CIFAR10_CLASSES[label]}) - poison pairs", fontsize=11)
    plt.tight_layout()
    out = Path(out_dir) / "poison_pairs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Step 3: victim ───────────────────────────────────────────────────────────
def train_victim(model, poisoned_dataset, epochs, batch_size, device):
    """Fine-tune the victim ResNet-18 on the poisoned training set (SGD + cosine LR)."""
    print("\n[Phase 3] Fine-tuning victim model on poisoned dataset...")
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(loader, desc=f"  Victim epoch {ep + 1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        sched.step()
        print(f"  epoch {ep + 1:2d}  loss={total_loss / len(loader):.4f}  train_acc={correct / total:.3f}")
    return model


# ── Step 4: evaluation (fixed averaged-delta "trigger") ──────────────────────
def extract_trigger(poisoned_dataset, train_dataset):
    """Mean pixel-space perturbation over all poisons -> one fixed trigger pattern."""
    deltas = []
    for idx in poisoned_dataset.poison_indices:
        x_orig = denormalize(train_dataset[idx][0]).cpu().numpy()
        x_poison = denormalize(poisoned_dataset.poisoned_images[idx]).cpu().numpy()
        deltas.append(x_poison - x_orig)
    return torch.tensor(np.mean(deltas, axis=0), dtype=torch.float32)


def apply_trigger(x, trigger, epsilon=None):
    x_01 = denormalize(x).clone()
    x_poison = x_01 + trigger
    if epsilon is not None:
        x_poison = x_01 + torch.clamp(x_poison - x_01, -epsilon, epsilon)
    return renormalize(torch.clamp(x_poison, 0, 1))


def evaluate(victim, test_dataset, trigger, base_class, target_class, batch_size, device, n_asr_samples=200):
    """CDA on the full test set; ASR = fraction of (200 random) triggered BASE-class test images
    that the victim classifies as the TARGET class.

    NB: this ASR definition differs from the Phase 2 definition (see docs/reproducibility_notes.md).
    """
    print("\n[Phase 4] Evaluation (fixed averaged trigger)")
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    victim.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (victim(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    cda = correct / total
    print(f"CDA = {cda:.4f}")

    base_idx = class_indices(test_dataset, base_class)
    sample_idx = random.sample(base_idx, min(n_asr_samples, len(base_idx)))
    success = 0
    for idx in sample_idx:
        x_trig = apply_trigger(test_dataset[idx][0], trigger)
        with torch.no_grad():
            pred = victim(x_trig.unsqueeze(0).to(device)).argmax(1).item()
        success += int(pred == target_class)
    asr = success / len(sample_idx)
    print(f"ASR = {asr:.4f}")
    return cda, asr
