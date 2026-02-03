import inspect
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from models_pytorch.utils import DS_Combin


def _to_numpy(array):
    """Convert torch tensors to numpy arrays while leaving numpy arrays untouched."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _one_hot_targets(targets):
    """Normalize targets to class indices (handles one-hot targets)."""
    targets = _to_numpy(targets)
    if targets.ndim == 1:
        return targets
    return np.argmax(targets, axis=1)


def _accepts_single_input(model):
    """Return True if the model forward expects a single input tensor."""
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    params = list(signature.parameters.values())
    return len(params) == 2


def _prepare_single_input(x_tensor):
    """Prepare tensor for models that only accept X (e.g., plain MNL)."""
    if x_tensor.ndim == 4 and x_tensor.shape[-1] == 1:
        return x_tensor.squeeze(-1)
    return x_tensor


def get_probabilities(model, x_data, q_data, evidential=False):
    """Run the model and return class probabilities plus evidential outputs if requested."""
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x_tensor = torch.tensor(x_data).float().to(device)
        q_tensor = torch.tensor(q_data).long().to(device)
        single_input = _accepts_single_input(model)
        if single_input:
            x_tensor = _prepare_single_input(x_tensor)
        if evidential:
            evidence = model(x_tensor) if single_input else model(x_tensor, q_tensor)
            alpha = {key: value + 1 for key, value in evidence.items()}
            alpha_a = DS_Combin(alpha, evidence[0].shape[1])
            probs = alpha_a / torch.sum(alpha_a, dim=1, keepdim=True)
            return probs.cpu(), {k: v.cpu() for k, v in evidence.items()}, alpha_a.cpu()
        logits = model(x_tensor) if single_input else model(x_tensor, q_tensor)
        return torch.softmax(logits, dim=1).cpu(), None, None


def classification_metrics(probs, targets):
    """Compute standard classification metrics and confusion matrix."""
    probs_np = _to_numpy(probs)
    true_labels = _one_hot_targets(targets)
    pred_labels = np.argmax(probs_np, axis=1)
    acc = float(np.mean(pred_labels == true_labels))
    f1 = f1_score(true_labels, pred_labels, average="macro")
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    matrix = confusion_matrix(true_labels, pred_labels)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix,
    }


def expected_calibration_error(probs, targets, n_bins=15):
    """Estimate Expected Calibration Error (ECE) via confidence binning."""
    probs_np = _to_numpy(probs)
    true_labels = _one_hot_targets(targets)
    confidences = np.max(probs_np, axis=1)
    predictions = np.argmax(probs_np, axis=1)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_start, bin_end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > bin_start) & (confidences <= bin_end)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(predictions[mask] == true_labels[mask])
        bin_confidence = np.mean(confidences[mask])
        ece += np.abs(bin_accuracy - bin_confidence) * np.mean(mask)
    return float(ece)


def brier_score(probs, targets):
    """Compute the multiclass Brier score for probabilistic predictions."""
    probs_np = _to_numpy(probs)
    targets_np = _to_numpy(targets)
    if targets_np.ndim == 1:
        num_classes = probs_np.shape[1]
        targets_np = np.eye(num_classes)[targets_np]
    return float(np.mean(np.sum((probs_np - targets_np) ** 2, axis=1)))


def extract_exog_betas(model):
    """Extract exogenous coefficients (beta) if the model exposes them."""
    if hasattr(model, "utilities2"):
        betas = model.utilities2.weight.detach().cpu().numpy().flatten()
        return betas
    if hasattr(model, "conv"):
        betas = model.conv.weight.detach().cpu().numpy().flatten()
        return betas
    if hasattr(model, "beta"):
        betas = model.beta.detach().cpu().mean(dim=1).numpy().flatten()
        return betas
    return None


def compute_direct_elasticities(probs, x_data, betas_exog, x_vars):
    """Compute direct elasticities for each continuous variable."""
    if betas_exog is None:
        return {}
    probs_np = _to_numpy(probs)
    x_np = _to_numpy(x_data).squeeze(-1)
    elasticities = {}
    for idx, var in enumerate(x_vars):
        beta = betas_exog[idx]
        elasticity = beta * x_np[:, idx, :] * (1.0 - probs_np)
        elasticities[var] = {
            "mean": float(np.mean(elasticity)),
            "mean_abs": float(np.mean(np.abs(elasticity))),
            "by_choice": np.mean(elasticity, axis=0).tolist(),
        }
    return elasticities


def compute_vot(
    betas_exog,
    x_vars,
    time_var="TT_SCALED(/100)",
    cost_var="COST_SCALED(/100)",
    return_components=False,
):
    """Compute Value of Time (VoT) from time/cost coefficients when available."""
    if betas_exog is None:
        return None
    if time_var not in x_vars or cost_var not in x_vars:
        return None
    beta_time = betas_exog[x_vars.index(time_var)]
    beta_cost = betas_exog[x_vars.index(cost_var)]
    if beta_cost == 0:
        return None
    vot = float(-beta_time / beta_cost)
    if return_components:
        return {
            "vot": vot,
            "beta_time": float(beta_time),
            "beta_cost": float(beta_cost),
        }
    return vot


def compute_uncertainty(alpha):
    """Compute evidential uncertainty from Dirichlet alpha parameters."""
    if alpha is None:
        return None
    alpha_np = _to_numpy(alpha)
    num_choices = alpha_np.shape[1]
    return num_choices / np.sum(alpha_np, axis=1)


def evaluate_model(model, x_data, q_data, y_data, x_vars=None, evidential=False, n_bins=15):
    """Evaluate a model and return a dictionary of classification/calibration metrics."""
    probs, evidence, alpha = get_probabilities(model, x_data, q_data, evidential=evidential)
    metrics = classification_metrics(probs, y_data)
    metrics["ece"] = expected_calibration_error(probs, y_data, n_bins=n_bins)
    metrics["brier_score"] = brier_score(probs, y_data)
    betas_exog = extract_exog_betas(model)
    if x_vars:
        metrics["elasticities"] = compute_direct_elasticities(probs, x_data, betas_exog, x_vars)
        metrics["vot"] = compute_vot(betas_exog, x_vars, return_components=True)
    else:
        metrics["elasticities"] = {}
        metrics["vot"] = None
    metrics["uncertainty"] = compute_uncertainty(alpha)
    metrics["evidence"] = evidence
    metrics["alpha"] = alpha
    return metrics
