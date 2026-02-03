from SM_data import (
    X_TRAIN,
    Q_TRAIN,
    y_TRAIN,
    X_TEST,
    Q_TEST,
    y_TEST,
    X_vars,
)
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import (
    E_MNL_train,
    EL_MNL_train,
    L_MNL_train,
    TE_MNL_train,
    TEL_MNL_train,
)
from models_pytorch.metrics import evaluate_model


def print_report(name, metrics):
    print(f"=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"VoT: {metrics['vot']}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    if metrics["elasticities"]:
        print("Elasticities (mean_abs):")
        for var, stats in metrics["elasticities"].items():
            print(f"  {var}: {stats['mean_abs']:.4f}")
    print()


def main():
    n_epochs = 200
    lr = 0.001
    l2 = 0.00001
    batch_size = 100
    drop = 0.2
    extra_emb_dims = 2
    n_nodes = 15
    lambda_epochs = 100
    configs = [
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

    for config in configs:
        trainer = config["trainer"]
        model_cls = config["model"]
        trained_model, *_ = trainer(
            X_TRAIN,
            Q_TRAIN,
            y_TRAIN,
            model_cls,
            N_EPOCHS=n_epochs,
            LR=lr,
            l2=l2,
            BATCH_SIZE=batch_size,
            VERBOSE=0,
            save_model=0,
            **config["kwargs"],
        )
        metrics = evaluate_model(
            trained_model,
            X_TEST,
            Q_TEST,
            y_TEST,
            x_vars=X_vars,
            evidential=config["evidential"],
        )
        print_report(config["name"], metrics)


if __name__ == "__main__":
    main()
