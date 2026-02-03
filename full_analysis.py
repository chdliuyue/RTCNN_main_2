import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars
from models_pytorch.models import MNL, E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import (
    MNL_train,
    E_MNL_train,
    EL_MNL_train,
    L_MNL_train,
    TE_MNL_train,
    TEL_MNL_train,
)
from models_pytorch.metrics import evaluate_model
from models_pytorch.analysis import (
    run_tel_mnl_sensitivity,
    plot_embedding_dim_sensitivity,
    run_tel_mnl_hyperparameter_ablation,
    uncertainty_noise_analysis,
    plot_noise_uncertainty_trend,
    run_tel_mnl_ablation,
    plot_tel_mnl_ablation,
    extract_parameters_and_uncertainty,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def save_json(data, path: Path):
    safe = {k: _json_safe(v) for k, v in data.items()}
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False))


def plot_confusion_matrix(matrix, output_path: Path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_elasticities(elasticities, output_path: Path):
    if not elasticities:
        return
    rows = []
    for var, stats in elasticities.items():
        row = {
            "variable": var,
            "mean": stats["mean"],
            "mean_abs": stats["mean_abs"],
        }
        for idx, value in enumerate(stats["by_choice"]):
            row[f"choice_{idx}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_uncertainty(uncertainty, output_dir: Path, prefix: str):
    if uncertainty is None:
        return None
    uncertainty = np.asarray(uncertainty)
    df = pd.DataFrame({"uncertainty": uncertainty})
    csv_path = output_dir / f"{prefix}_uncertainty.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(uncertainty, bins=30, color="tab:purple", alpha=0.7)
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix} Uncertainty Distribution")
    fig.tight_layout()
    hist_path = output_dir / f"{prefix}_uncertainty_hist.png"
    fig.savefig(hist_path)
    plt.close(fig)
    return {
        "uncertainty_csv": str(csv_path),
        "uncertainty_hist": str(hist_path),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "std_uncertainty": float(np.std(uncertainty)),
    }


def save_metrics_bundle(model_name, split_name, metrics, output_dir: Path):
    prefix = f"{model_name.lower()}_{split_name.lower()}"
    summary = {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "ece": metrics["ece"],
        "brier_score": metrics["brier_score"],
        "vot": metrics["vot"],
    }

    summary_path = output_dir / f"{prefix}_metrics.json"
    save_json(summary, summary_path)

    confusion_matrix = metrics["confusion_matrix"]
    pd.DataFrame(confusion_matrix).to_csv(output_dir / f"{prefix}_confusion_matrix.csv", index=False)
    plot_confusion_matrix(
        confusion_matrix,
        output_dir / f"{prefix}_confusion_matrix.png",
        title=f"{model_name} ({split_name}) Confusion Matrix",
    )

    save_elasticities(metrics["elasticities"], output_dir / f"{prefix}_elasticities.csv")
    uncertainty_info = save_uncertainty(metrics["uncertainty"], output_dir, prefix)
    if uncertainty_info:
        summary.update(uncertainty_info)
        save_json(summary, summary_path)


def run_model_suite(output_dir: Path):
    n_epochs = 200
    lr = 0.001
    l2 = 0.00001
    batch_size = 100
    drop = 0.2
    extra_emb_dims = 2
    n_nodes = 15
    lambda_epochs = 100

    configs = [
        {"name": "MNL", "trainer": MNL_train, "model": MNL, "kwargs": {}, "evidential": False},
        {"name": "E_MNL", "trainer": E_MNL_train, "model": E_MNL, "kwargs": {"drop": drop}, "evidential": False},
        {
            "name": "EL_MNL",
            "trainer": EL_MNL_train,
            "model": EL_MNL,
            "kwargs": {"extra_emb_dims": extra_emb_dims, "n_nodes": n_nodes, "drop": drop},
            "evidential": False,
        },
        {"name": "L_MNL", "trainer": L_MNL_train, "model": L_MNL, "kwargs": {"n_nodes": n_nodes}, "evidential": False},
        {
            "name": "TE_MNL",
            "trainer": TE_MNL_train,
            "model": TE_MNL,
            "kwargs": {"lambda_epochs": lambda_epochs, "drop": drop},
            "evidential": True,
        },
        {
            "name": "TEL_MNL",
            "trainer": TEL_MNL_train,
            "model": TEL_MNL,
            "kwargs": {
                "lambda_epochs": lambda_epochs,
                "extra_emb_dims": extra_emb_dims,
                "n_nodes": n_nodes,
                "drop": drop,
            },
            "evidential": True,
        },
    ]

    summary_rows = []
    for config in configs:
        trained_model, *_ = config["trainer"](
            X_TRAIN,
            Q_TRAIN,
            y_TRAIN,
            config["model"],
            N_EPOCHS=n_epochs,
            LR=lr,
            l2=l2,
            BATCH_SIZE=batch_size,
            VERBOSE=0,
            save_model=0,
            **config["kwargs"],
        )

        metrics_train = evaluate_model(
            trained_model,
            X_TRAIN,
            Q_TRAIN,
            y_TRAIN,
            x_vars=X_vars,
            evidential=config["evidential"],
        )
        metrics_test = evaluate_model(
            trained_model,
            X_TEST,
            Q_TEST,
            y_TEST,
            x_vars=X_vars,
            evidential=config["evidential"],
        )

        save_metrics_bundle(config["name"], "train", metrics_train, output_dir)
        save_metrics_bundle(config["name"], "test", metrics_test, output_dir)

        summary_rows.append(
            {
                "model": config["name"],
                "train_accuracy": metrics_train["accuracy"],
                "test_accuracy": metrics_test["accuracy"],
                "train_f1": metrics_train["f1"],
                "test_f1": metrics_test["f1"],
                "train_ece": metrics_train["ece"],
                "test_ece": metrics_test["ece"],
                "train_brier": metrics_train["brier_score"],
                "test_brier": metrics_test["brier_score"],
            }
        )

        if config["name"] == "TEL_MNL":
            betas, uncertainty = extract_parameters_and_uncertainty(
                trained_model, X_TEST, Q_TEST, y_TEST
            )
            if betas is not None:
                pd.DataFrame({"variable": X_vars, "beta": betas}).to_csv(
                    output_dir / "tel_mnl_betas.csv", index=False
                )
            if uncertainty is not None:
                pd.DataFrame({"uncertainty": uncertainty}).to_csv(
                    output_dir / "tel_mnl_uncertainty_values.csv", index=False
                )

    pd.DataFrame(summary_rows).to_csv(output_dir / "model_summary.csv", index=False)


def run_tel_mnl_analyses(output_dir: Path):
    lambda_epochs = 100
    extra_emb_dims_values = [1, 2, 3, 4]
    n_nodes_values = [10, 15, 20]

    sensitivity_results = run_tel_mnl_sensitivity(
        X_TRAIN,
        Q_TRAIN,
        y_TRAIN,
        X_TEST,
        Q_TEST,
        y_TEST,
        TEL_MNL,
        lambda_epochs=lambda_epochs,
        extra_emb_dims_values=extra_emb_dims_values,
        n_nodes_values=n_nodes_values,
        n_epochs=200,
        lr=0.001,
        l2=0.00001,
        batch_size=100,
        drop=0.2,
    )
    sensitivity_results.to_csv(output_dir / "tel_mnl_sensitivity_results.csv", index=False)
    plot_embedding_dim_sensitivity(
        sensitivity_results,
        metric="accuracy",
        output_path=output_dir / "tel_mnl_embedding_sensitivity.png",
    )
    run_tel_mnl_hyperparameter_ablation(
        sensitivity_results,
        output_path=output_dir / "tel_mnl_hyperparameter_sensitivity.png",
    )

    ablation_results = run_tel_mnl_ablation(
        X_TRAIN,
        Q_TRAIN,
        y_TRAIN,
        X_TEST,
        Q_TEST,
        y_TEST,
        TEL_MNL,
        lambda_epochs=lambda_epochs,
        extra_emb_dims=2,
        n_nodes=15,
        n_epochs=200,
        lr=0.001,
        l2=0.00001,
        batch_size=100,
        drop=0.2,
    )
    ablation_results.to_csv(output_dir / "tel_mnl_ablation_results.csv", index=False)
    plot_tel_mnl_ablation(ablation_results, output_path=output_dir / "tel_mnl_ablation.png")

    trained_model, *_ = TEL_MNL_train(
        X_TRAIN,
        Q_TRAIN,
        y_TRAIN,
        TEL_MNL,
        lambda_epochs=lambda_epochs,
        extra_emb_dims=2,
        n_nodes=15,
        N_EPOCHS=200,
        LR=0.001,
        l2=0.00001,
        BATCH_SIZE=100,
        drop=0.2,
        VERBOSE=0,
        save_model=0,
    )
    noise_results = uncertainty_noise_analysis(
        trained_model,
        X_TEST,
        Q_TEST,
        y_TEST,
        noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    noise_results.to_csv(output_dir / "tel_mnl_uncertainty_noise.csv", index=False)
    plot_noise_uncertainty_trend(noise_results, output_path=output_dir / "tel_mnl_uncertainty_noise.png")


def main():
    output_dir = _ensure_dir(Path("analysis_outputs"))
    run_model_suite(output_dir)
    run_tel_mnl_analyses(output_dir)


if __name__ == "__main__":
    main()
