"""One poisoned-victim experiment: train (or reuse) the clean victim, train the poisoned victim, evaluate."""
import torch

from diffusion_poison import PoisonedCIFAR10
from shared_utils.data import CIFAR10_CLASSES
from shared_utils.metrics import (check_target_misclassified, evaluate_clean_accuracy,
                                  evaluate_target_class_asr, get_confusion_matrix)
from shared_utils.plotting import plot_confusion_matrix
from shared_utils.training import train_victim


def run_poisoned_experiment(poisons, target_img, target_idx, train_set, test_set, victim_clean, device,
                            base_class_idx, target_class_idx, arch="surrogate", victim_epochs=8,
                            batch_size=256, fig_prefix=None):
    """Returns (result_dict, victim_poisoned). Confusion-matrix PNGs are written if fig_prefix is given."""
    poisoned_ds = PoisonedCIFAR10(train_set, poisons)
    victim_poisoned = train_victim(poisoned_ds, device, arch=arch, epochs=victim_epochs, batch_size=batch_size)

    cda_clean = evaluate_clean_accuracy(victim_clean, test_set, device)
    cda_pois = evaluate_clean_accuracy(victim_poisoned, test_set, device)
    succ_before, pred_before = check_target_misclassified(victim_clean, target_img, base_class_idx, device)
    succ_after, pred_after = check_target_misclassified(victim_poisoned, target_img, base_class_idx, device)
    asr_c, n_c, n_tot = evaluate_target_class_asr(victim_clean, test_set, target_class_idx, base_class_idx, device)
    asr_p, n_p, _ = evaluate_target_class_asr(victim_poisoned, test_set, target_class_idx, base_class_idx, device)

    if fig_prefix:
        for name, model in (("clean", victim_clean), ("poisoned", victim_poisoned)):
            cm = get_confusion_matrix(model, test_set, device)
            plot_confusion_matrix(cm, CIFAR10_CLASSES, f"Confusion Matrix - {name.capitalize()} Model",
                                  save_path=f"{fig_prefix}_confusion_{name}.png")

    result = {
        "arch": arch, "n_poisons": len(poisons), "poison_rate": len(poisons) / len(train_set),
        "clean_trained_acc": cda_clean, "poison_trained_acc": cda_pois,
        "clean_model_asr": asr_c, "poisoned_model_asr": asr_p, "asr_increase_pp": (asr_p - asr_c) * 100,
        "clean_model_asr_counts": [n_c, n_tot], "poisoned_model_asr_counts": [n_p, n_tot],
        "single_target": {"target_idx": target_idx,
                          "clean_pred": CIFAR10_CLASSES[pred_before], "poisoned_pred": CIFAR10_CLASSES[pred_after],
                          "success_before": bool(succ_before), "success_after": bool(succ_after)},
    }
    print(f"CDA clean/poisoned: {cda_clean:.4f}/{cda_pois:.4f} | "
          f"ASR clean {asr_c * 100:.2f}% -> poisoned {asr_p * 100:.2f}% (+{(asr_p - asr_c) * 100:.2f} pp)")
    return result, victim_poisoned


def load_or_train_clean_victim(train_set, device, arch, epochs, batch_size, ckpt_path=None):
    """Clean baseline victim; reuses a checkpoint if one exists at ckpt_path."""
    from pathlib import Path

    from shared_utils.models import build_model

    if ckpt_path and Path(ckpt_path).exists():
        model = build_model(arch).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model.eval()
    model = train_victim(train_set, device, arch=arch, epochs=epochs, batch_size=batch_size)
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    return model
