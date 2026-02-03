from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import E_MNL_train, EL_MNL_train, L_MNL_train, TE_MNL_train, TEL_MNL_train
from models_pytorch.metrics import evaluate_model


N_EPOCHS = 200
LR = 0.001
l2 = 0.00001
BATCH_SIZE = 100
drop = 0.2
VERBOSE = 1
save_model = 0

extra_emb_dims = 2
n_nodes = 15

lambda_epochs = 100


def _print_elasticities(elasticities):
    """Print elasticity statistics when available."""
    if not elasticities:
        print("Elasticities: N/A")
        return
    print("Elasticities (mean_abs | mean):")
    for var, stats in elasticities.items():
        print(f"  {var}: {stats['mean_abs']:.4f} | {stats['mean']:.4f}")


def _print_report(model_name, split_name, metrics):
    """Pretty-print the full metric report for a model/split."""
    print(f"=== {model_name} ({split_name}) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"VoT: {metrics['vot']}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    _print_elasticities(metrics["elasticities"])
    print()


CONFIGS = [
    {
        "name": "E_MNL",
        "trainer": E_MNL_train,
        "model": E_MNL,
        "kwargs": {"drop": drop},
        "evidential": False,
    },
    {
        "name": "EL_MNL",
        "trainer": EL_MNL_train,
        "model": EL_MNL,
        "kwargs": {"extra_emb_dims": extra_emb_dims, "n_nodes": n_nodes, "drop": drop},
        "evidential": False,
    },
    {
        "name": "L_MNL",
        "trainer": L_MNL_train,
        "model": L_MNL,
        "kwargs": {"n_nodes": n_nodes},
        "evidential": False,
    },
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


for config in CONFIGS:
    # Train each model once and report both train/test metrics.
    trained_model, *_ = config["trainer"](
        X_TRAIN,
        Q_TRAIN,
        y_TRAIN,
        config["model"],
        N_EPOCHS=N_EPOCHS,
        LR=LR,
        l2=l2,
        BATCH_SIZE=BATCH_SIZE,
        VERBOSE=VERBOSE,
        save_model=save_model,
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
    _print_report(config["name"], "train", metrics_train)
    _print_report(config["name"], "test", metrics_test)
