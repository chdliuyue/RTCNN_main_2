import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models_pytorch.trainer import TEL_MNL_train
from models_pytorch.metrics import evaluate_model, compute_uncertainty


def run_tel_mnl_sensitivity(X_train, Q_train, y_train, X_eval, Q_eval, y_eval, model,
                            lambda_epochs, extra_emb_dims_values, n_nodes_values,
                            n_epochs=200, lr=0.001, l2=0.0, batch_size=100, drop=0.2):
    results = []
    for extra_emb_dims in extra_emb_dims_values:
        for n_nodes in n_nodes_values:
            trained_model, *_ = TEL_MNL_train(
                X_train, Q_train, y_train, model,
                lambda_epochs=lambda_epochs,
                extra_emb_dims=extra_emb_dims,
                n_nodes=n_nodes,
                N_EPOCHS=n_epochs,
                LR=lr,
                l2=l2,
                BATCH_SIZE=batch_size,
                drop=drop,
                VERBOSE=0,
                save_model=0,
            )
            metrics = evaluate_model(
                trained_model,
                X_eval,
                Q_eval,
                y_eval,
                evidential=True,
            )
            results.append({
                "extra_emb_dims": extra_emb_dims,
                "n_nodes": n_nodes,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ece": metrics["ece"],
                "brier_score": metrics["brier_score"],
            })
    return pd.DataFrame(results)


def plot_embedding_dim_sensitivity(results_df, metric="accuracy", output_path="tel_mnl_embedding_sensitivity.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = results_df.groupby("extra_emb_dims")[metric].mean()
    ax.plot(grouped.index, grouped.values, marker="o")
    ax.set_xlabel("extra_emb_dims")
    ax.set_ylabel(metric)
    ax.set_title(f"TEL_MNL Embedding Dimension Sensitivity ({metric})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path


def run_tel_mnl_hyperparameter_ablation(results_df, output_path="tel_mnl_hyperparameter_sensitivity.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_nodes, subset in results_df.groupby("n_nodes"):
        ax.plot(
            subset["extra_emb_dims"],
            subset["accuracy"],
            marker="o",
            label=f"n_nodes={n_nodes}",
        )
    ax.set_xlabel("extra_emb_dims")
    ax.set_ylabel("accuracy")
    ax.set_title("TEL_MNL Hyperparameter Sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path


def uncertainty_noise_analysis(model, X_data, Q_data, y_data, noise_levels, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for std in noise_levels:
        noise = rng.normal(0, std, size=X_data.shape)
        noisy_x = X_data + noise
        metrics = evaluate_model(model, noisy_x, Q_data, y_data, evidential=True)
        mean_uncertainty = None
        if metrics["alpha"] is not None:
            mean_uncertainty = float(np.mean(compute_uncertainty(metrics["alpha"])))
        rows.append({
            "noise_std": std,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "mean_uncertainty": mean_uncertainty,
        })
    return pd.DataFrame(rows)


def extract_parameters_and_uncertainty(model, X_data, Q_data, y_data):
    metrics = evaluate_model(model, X_data, Q_data, y_data, evidential=True)
    betas = None
    if hasattr(model, "utilities2"):
        betas = model.utilities2.weight.detach().cpu().numpy().flatten()
    uncertainty = metrics["uncertainty"]
    return betas, uncertainty
