"""Linear probe for binary safety classification on LLM activations."""

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_probe(train_X, train_y, input_dim=2304, lr=1e-3, weight_decay=0.01,
                epochs=50, device="cpu"):
    """Train a linear probe with BCE loss + L2 regularization.

    Returns:
        Trained probe (on CPU), list of (epoch, loss) tuples.
    """
    probe = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X = train_X.to(device)
    y = train_y.float().to(device)

    loss_log = []
    for epoch in range(1, epochs + 1):
        logits = probe(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            loss_val = loss.item()
            loss_log.append((epoch, loss_val))
            print(f"  Epoch {epoch:>3d}/{epochs}  loss={loss_val:.4f}")

    probe.cpu()
    return probe, loss_log


def evaluate_probe(probe, test_X, test_y, device="cpu"):
    """Evaluate probe and return metrics dict + print report."""
    probe.to(device).eval()
    X = test_X.to(device)
    y_true = test_y.numpy()

    with torch.no_grad():
        logits = probe(X)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_pred = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, probs),
    }

    print("\n===== Test Set Evaluation =====")
    for k, v in metrics.items():
        print(f"  {k:<12s}: {v:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"              Unsafe  Safe")
    print(f"  Unsafe      {cm[0][0]:>5d}  {cm[0][1]:>5d}")
    print(f"  Safe        {cm[1][0]:>5d}  {cm[1][1]:>5d}")

    acc = metrics["accuracy"]
    if acc < 0.60:
        print(f"\n  WARNING: Accuracy {acc:.2%} is suspiciously low (<60%).")
    elif acc > 0.95:
        print(f"\n  WARNING: Accuracy {acc:.2%} is suspiciously high (>95%).")

    probe.cpu()
    return metrics
