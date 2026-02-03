from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST
from models_pytorch.models import TEL_MNL
from models_pytorch.analysis import (
    run_tel_mnl_sensitivity,
    plot_embedding_dim_sensitivity,
    run_tel_mnl_hyperparameter_ablation,
    uncertainty_noise_analysis,
)
from models_pytorch.trainer import TEL_MNL_train


def main():
    lambda_epochs = 100
    extra_emb_dims_values = [1, 2, 3, 4]
    n_nodes_values = [10, 15, 20]

    results = run_tel_mnl_sensitivity(
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
    results.to_csv("tel_mnl_sensitivity_results.csv", index=False)
    plot_embedding_dim_sensitivity(results, metric="accuracy", output_path="tel_mnl_embedding_sensitivity.png")
    run_tel_mnl_hyperparameter_ablation(results, output_path="tel_mnl_hyperparameter_sensitivity.png")

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
    noise_results.to_csv("tel_mnl_uncertainty_noise.csv", index=False)


if __name__ == "__main__":
    main()
