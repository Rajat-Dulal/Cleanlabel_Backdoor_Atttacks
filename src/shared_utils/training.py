import torch
import torch.nn.functional as F

from .models import SurrogateResNet, build_model


def train_classifier(model, dataset, device, epochs, batch_size, lr=1e-3, tag="Model", num_workers=2):
    """Plain supervised training: Adam, cross-entropy, no augmentation (as in the notebooks)."""
    model = model.to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"[{tag}] epoch {ep + 1}/{epochs} loss={loss_sum / total:.4f} acc={correct / total:.4f}")
    model.eval()
    return model


def train_surrogate(train_set, device, epochs=15, batch_size=128):
    """Guidance surrogate (attacker side): SurrogateResNet trained on full CIFAR-10."""
    return train_classifier(SurrogateResNet(), train_set, device, epochs, batch_size, tag="Surrogate")


def train_victim(dataset, device, arch="surrogate", epochs=8, batch_size=256):
    """Victim (defender side). arch='surrogate' = same-architecture; arch='resnet' = transfer setting."""
    return train_classifier(build_model(arch), dataset, device, epochs, batch_size, tag="Victim")
