"""SIT746 artefact: single-stage guided reverse diffusion for clean-label poison generation.

Algorithm 1 in the paper:  SDEdit-style partial forward diffusion of a base image to t_start, then reverse
denoising with a pretrained, frozen DDPM (google/ddpm-cifar10-32).  At every reverse step the predicted clean
image x0_hat is scored by
        L = MSE(phi(x0_hat), phi(x_target)) + lpips_weight * LPIPS(x0_hat, x_base)
and x_{t-1} is nudged against grad_{x_t} L, scaled by guidance_scale * t/1000.

NOTE: the notebooks use F.mse_loss (a MEAN over the 256 feature dims); the paper writes a squared L2 norm
(a SUM).  The two differ by a constant factor of 256, which is absorbed into `guidance_scale`.
All images are in [-1, 1].  Poisons keep the BASE class label (clean-label).
"""
import random

import torch
import torch.nn.functional as F

from shared_utils.data import class_indices, denorm_diffusion

try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:  # pragma: no cover
    _ssim = None


class DiffusionPoisoner:
    def __init__(self, device, model_id="google/ddpm-cifar10-32"):
        import lpips
        from diffusers import DDPMPipeline

        self.device = device
        pipe = DDPMPipeline.from_pretrained(model_id).to(device)
        self.unet = pipe.unet.eval().requires_grad_(False)   # frozen; we only need grad w.r.t. x_t
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(1000)
        self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    def make_poison(self, base_img, target_img, surrogate, t_start=100, guidance_scale=20.0,
                    lpips_weight=0.15, num_inference_steps=50):
        """base_img/target_img: [1,3,32,32] in [-1,1]. Returns poison [1,3,32,32] in [-1,1]."""
        dev = self.device
        base_img, target_img = base_img.to(dev), target_img.to(dev)
        with torch.no_grad():
            _, target_feat = surrogate(target_img, return_features=True)

        # 1) partial forward diffusion (SDEdit)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps                       # descending, e.g. [980, 960, ..., 0]
        start_idx = (timesteps - t_start).abs().argmin().item()
        t0 = timesteps[start_idx]
        x_t = self.scheduler.add_noise(base_img, torch.randn_like(base_img), t0.unsqueeze(0))

        # 2) guided reverse diffusion from t0 down to 0
        x_t = x_t.detach().clone().requires_grad_(True)
        for t in timesteps[start_idx:]:
            noise_pred = self.unet(x_t, t.unsqueeze(0).to(dev)).sample
            step_out = self.scheduler.step(noise_pred, t.item(), x_t)
            x_prev, pred_x0 = step_out.prev_sample, step_out.pred_original_sample

            pred_x0 = pred_x0.clamp(-1, 1)
            _, feat = surrogate(pred_x0, return_features=True)
            collision = F.mse_loss(feat, target_feat.detach())
            perceptual = self.lpips_fn(pred_x0, base_img).mean()
            total_loss = collision + lpips_weight * perceptual    # LPIPS acts as a penalty

            grad = torch.autograd.grad(total_loss, x_t, retain_graph=False)[0]
            with torch.no_grad():
                x_prev = x_prev - guidance_scale * (t.item() / 1000.0) * grad
            x_t = x_prev.detach().clone().requires_grad_(True)

        return x_t.detach().clamp(-1, 1)


def generate_poison_set(poisoner, surrogate, train_set, test_set, base_class_idx, target_class_idx,
                        num_poisons=25, t_start=250, guidance_scale=20.0, lpips_weight=0.15):
    """Pick `num_poisons` random base-class training images and one random target-class TEST image.

    Returns (poisons, target_img, target_idx) with poisons = [(train_idx, poison[1,3,32,32] cpu, base_label)].
    Uses the global `random` state -> call set_seed() first for reproducibility.
    """
    chosen_base = random.sample(class_indices(train_set, base_class_idx), num_poisons)
    target_idx = random.choice(class_indices(test_set, target_class_idx))
    target_img = test_set[target_idx][0].unsqueeze(0)

    poisons = []
    for i, idx in enumerate(chosen_base):
        base_img, base_label = train_set[idx]
        poison = poisoner.make_poison(base_img.unsqueeze(0), target_img, surrogate, t_start=t_start,
                                      guidance_scale=guidance_scale, lpips_weight=lpips_weight)
        poisons.append((idx, poison.cpu(), base_label))
        print(f"generated poison {i + 1}/{num_poisons}")
    return poisons, target_img, target_idx


class PoisonedCIFAR10(torch.utils.data.Dataset):
    """CIFAR-10 where the chosen training samples are REPLACED by their poisons (same index, same label).

    Dataset size is unchanged (50,000); poison rate = len(poisons) / 50,000.
    """

    def __init__(self, clean_dataset, poisons):
        self.clean_dataset = clean_dataset
        self.poisons = {idx: (img, label) for idx, img, label in poisons}

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        if idx in self.poisons:
            img, label = self.poisons[idx]
            return (img.squeeze(0) if img.dim() == 4 else img), label
        return self.clean_dataset[idx]


def compute_ssim_scores(poisons, train_set):
    """SSIM between each poison and its original clean training image (None if scikit-image missing)."""
    if _ssim is None:
        return None

    def to_np(t):
        t = t[0] if t.dim() == 4 else t
        return denorm_diffusion(t).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    return [float(_ssim(to_np(train_set[idx][0]), to_np(p), channel_axis=2, data_range=1.0))
            for idx, p, _ in poisons]


def save_poisons(path, poisons, target_img, target_idx, meta):
    torch.save({"poisons": poisons, "target_img": target_img, "target_idx": target_idx, "meta": meta}, path)


def load_poisons(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["poisons"], blob["target_img"], blob["target_idx"], blob["meta"]
