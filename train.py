import numpy as np
from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import E_MNL_train, EL_MNL_train, L_MNL_train, TE_MNL_train, TEL_MNL_train
from models_pytorch.metrics import evaluate_model


N_EPOCHS = 200
LR = 0.001
l2 = 0.00001
BATCH_SIZE = 100
drop = 0.2  # 0
VERBOSE = 1
save_model = 0

extra_emb_dims = 2
n_nodes = 15  # 15

lambda_epochs = 100
train_acc, train_f1 = [], []
test_acc, test_f1 = [], []
MODEL_NAME = "EL_MNL"
MODEL_CONFIGS = {
    "E_MNL": {
        "trainer": E_MNL_train,
        "model": E_MNL,
        "kwargs": {"drop": drop},
    },
    "EL_MNL": {
        "trainer": EL_MNL_train,
        "model": EL_MNL,
        "kwargs": {"extra_emb_dims": extra_emb_dims, "n_nodes": n_nodes, "drop": drop},
    },
    "L_MNL": {
        "trainer": L_MNL_train,
        "model": L_MNL,
        "kwargs": {"n_nodes": n_nodes},
    },
    "TE_MNL": {
        "trainer": TE_MNL_train,
        "model": TE_MNL,
        "kwargs": {"lambda_epochs": lambda_epochs, "drop": drop},
    },
    "TEL_MNL": {
        "trainer": TEL_MNL_train,
        "model": TEL_MNL,
        "kwargs": {
            "lambda_epochs": lambda_epochs,
            "extra_emb_dims": extra_emb_dims,
            "n_nodes": n_nodes,
            "drop": drop,
        },
    },
}

for i in range(1):
    if MODEL_NAME not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_NAME={MODEL_NAME}. Options: {sorted(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[MODEL_NAME]
    trainer = config["trainer"]
    model_cls = config["model"]
    trained_model, *_ = trainer(
        X_TRAIN,
        Q_TRAIN,
        y_TRAIN,
        model_cls,
        N_EPOCHS=N_EPOCHS,
        LR=LR,
        l2=l2,
        BATCH_SIZE=BATCH_SIZE,
        VERBOSE=VERBOSE,
        save_model=save_model,
        **config["kwargs"],
    )

    evidential = MODEL_NAME in {"TE_MNL", "TEL_MNL"}
    metrics_train = evaluate_model(trained_model, X_TRAIN, Q_TRAIN, y_TRAIN, x_vars=X_vars, evidential=evidential)
    metrics_test = evaluate_model(trained_model, X_TEST, Q_TEST, y_TEST, x_vars=X_vars, evidential=evidential)
    train_acc.append(metrics_train["accuracy"])
    train_f1.append(metrics_train["f1"])
    test_acc.append(metrics_test["accuracy"])
    test_f1.append(metrics_test["f1"])

re_acc, st_acc = np.mean(train_acc), np.std(train_acc)
re_f1, st_f1 = np.mean(train_f1), np.std(train_f1)
print("######################################################")
print('train acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('train f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

re2_acc, st2_acc = np.mean(test_acc), np.std(test_acc)
re2_f1, st2_f1 = np.mean(test_f1), np.std(test_f1)
print("######################################################")
print('test acc: {:.3f}, std: {:.3f}'.format(re2_acc, st2_acc))
print('test f1: {:.3f}, std: {:.3f}'.format(re2_f1, st2_f1))


