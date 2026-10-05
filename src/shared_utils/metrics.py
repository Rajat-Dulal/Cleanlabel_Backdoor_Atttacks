import torch
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def evaluate_clean_accuracy(model, dataset, device, batch_size=512):
    """CDA: overall test accuracy."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def check_target_misclassified(model, target_img, base_class_idx, device):
    """Single-target check: is the one chosen target image predicted as the poison (base) class?"""
    model.eval()
    pred = model(target_img.to(device)).argmax(1).item()
    return pred == base_class_idx, pred


@torch.no_grad()
def evaluate_target_class_asr(model, dataset, target_class_idx, base_class_idx, device, batch_size=512):
    """Aggregate ASR used in the paper (Phase 2).

    Fraction of held-out test images of `target_class_idx` (dog) that the model classifies as
    `base_class_idx` (cat). Returns (asr, n_success, n_total).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    n_total, n_success = 0, 0
    for x, y in loader:
        mask = y == target_class_idx
        if not mask.any():
            continue
        pred = model(x[mask].to(device)).argmax(dim=1)
        n_success += (pred == base_class_idx).sum().item()
        n_total += int(mask.sum())
    return (n_success / n_total if n_total else 0.0), n_success, n_total


@torch.no_grad()
def get_confusion_matrix(model, dataset, device, num_classes=10, batch_size=512):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(dim=1).cpu().numpy())
        labels.extend(y.numpy())
    return confusion_matrix(labels, preds, labels=list(range(num_classes)))
