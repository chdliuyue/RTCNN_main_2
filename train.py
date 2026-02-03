import torch
import numpy as np
from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars, Q_vars, Q_df_TRAIN, cats2ints_mapping
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import E_MNL_train, EL_MNL_train, L_MNL_train, TE_MNL_train, TEL_MNL_train
from models_pytorch.utils import model_predict, t_model_predict
from models_pytorch.estimation import model_summary, get_betas_and_embeddings
import matplotlib.pyplot as plt
from adjustText import adjust_text


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
train_loss, train_acc, train_f1 = [], [], []
test_loss, test_acc, test_f1 = [], [], []
for i in range(1):
    trained_model, Loss, Acc, LL, f1 = E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
                                                   N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
                                                   VERBOSE=VERBOSE, save_model=save_model,
                                                   model_filename='e_mnl_model.pth')
    # trained_model, Loss, Acc, LL, f1 = L_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, L_MNL,
    #                                                n_nodes=n_nodes, N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE,
    #                                                VERBOSE=VERBOSE, save_model=save_model, model_filename='l_mnl_model.pth')
    trained_model, Loss, Acc, LL, f1 = EL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, EL_MNL,
                                                    extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
                                                    N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
                                                    VERBOSE=VERBOSE, save_model=save_model, model_filename='el_mnl_model.pth')
    # trained_model, Loss, Acc, f1 = TE_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TE_MNL, lambda_epochs=lambda_epochs,
    #                                             N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
    #                                             VERBOSE=VERBOSE, save_model=save_model, model_filename='te_mnl_model.pth')
    # trained_model, Loss, Acc, f1 = TEL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TEL_MNL, lambda_epochs=lambda_epochs,
    #                                             extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
    #                                             N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
    #                                             VERBOSE=VERBOSE, save_model=save_model, model_filename='te_mnl_model.pth')

    # E_MNL/L_MNL/EL_MNL
    # tr_loss, tr_acc, tr_ll, tr_f1, _ = model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model)
    # te_loss, te_acc, te_ll, te_f1, _ = model_predict(X_TEST, Q_TEST, y_TEST, trained_model)
    # TE_MNL/TEL_MNL
    tr_loss, tr_acc, tr_f1, _, _ = t_model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model, N_EPOCHS, lambda_epochs)
    te_loss, te_acc, te_f1, _, _ = t_model_predict(X_TEST, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)

    train_loss.append(tr_loss)
    train_acc.append(tr_acc)
    train_f1.append(tr_f1)
    test_loss.append(te_loss)
    test_acc.append(te_acc)
    test_f1.append(te_f1)

re_loss, st_loss = np.mean(train_loss), np.std(train_loss)
re_acc, st_acc = np.mean(train_acc), np.std(train_acc)
re_f1, st_f1 = np.mean(train_f1), np.std(train_f1)
print("######################################################")
print('train Loss: {:.3f}, std: {:.3f}'.format(re_loss, st_loss))
print('train acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('train f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

re2_loss, st2_loss = np.mean(test_loss), np.std(test_loss)
re2_acc, st2_acc = np.mean(test_acc), np.std(test_acc)
re2_f1, st2_f1 = np.mean(test_f1), np.std(test_f1)
print("######################################################")
print('test Loss: {:.3f}, std: {:.3f}'.format(re2_loss, st2_loss))
print('test acc: {:.3f}, std: {:.3f}'.format(re2_acc, st2_acc))
print('test f1: {:.3f}, std: {:.3f}'.format(re2_f1, st2_f1))







