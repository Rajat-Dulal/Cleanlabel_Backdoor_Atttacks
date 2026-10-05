import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

from .data import denorm_diffusion


def plot_confusion_matrix(cm, classes, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def save_poison_grid(poisons, target_img, classes, save_path, n=5):
    """Top row: first n poisons (with their retained label). Bottom row: the target test image."""
    fig, axes = plt.subplots(2, n, figsize=(12, 5))
    for i in range(min(n, len(poisons))):
        axes[0, i].imshow(denorm_diffusion(poisons[i][1][0]).permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"poison ({classes[poisons[i][2]]})")
        axes[0, i].axis("off")
    axes[1, n // 2].imshow(denorm_diffusion(target_img[0]).permute(1, 2, 0).numpy())
    axes[1, n // 2].set_title("target (test) image")
    for i in range(n):
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
