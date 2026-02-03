import torch
import numpy as np
from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars, Q_vars, Q_df_TRAIN, cats2ints_mapping
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import E_MNL_train, EL_MNL_train, L_MNL_train, TE_MNL_train, TEL_MNL_train
from models_pytorch.utils import model_predict, t_model_predict
from models_pytorch.estimation import model_summary, get_betas_and_embeddings
import matplotlib.pyplot as plt
from adjustText import adjust_text


# Experiment: evaluate robustness under Gaussian noise perturbations.
N_EPOCHS = 200
LR = 0.005
l2 = 0.00001
BATCH_SIZE = 100
drop = 1  # 0
VERBOSE = 1
save_model = 0

extra_emb_dims = 2
n_nodes = 15  # 15

lambda_epochs = 100

# Build noisy variants of the test set to measure degradation.
noise1 = np.random.normal(0, 0.2, size=X_TEST.shape)
noise2 = np.random.normal(0, 0.4, size=X_TEST.shape)
noise3 = np.random.normal(0, 0.6, size=X_TEST.shape)
noise4 = np.random.normal(0, 0.8, size=X_TEST.shape)


noise_X1 = X_TEST + noise1
noise_X2 = X_TEST + noise2
noise_X3 = X_TEST + noise3
noise_X4 = X_TEST + noise4


test_acc1, test_f1 = [], []
test_acc2, test_f2 = [], []
test_acc3, test_f3 = [], []
test_acc4, test_f4 = [], []

for i in range(5):
    trained_model, Loss, Acc, LL, f1 = E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
                                                   N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
                                                   VERBOSE=VERBOSE, save_model=save_model,
                                                   model_filename='e_mnl_model.pth')
    # trained_model, Loss, Acc, LL, f1 = L_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, L_MNL,
    #                                                n_nodes=n_nodes, N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE,
    #                                                VERBOSE=VERBOSE, save_model=save_model, model_filename='l_mnl_model.pth')
    # trained_model, Loss, Acc, LL, f1 = EL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, EL_MNL,
    #                                                 extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
    #                                                 N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
    #                                                 VERBOSE=VERBOSE, save_model=save_model, model_filename='el_mnl_model.pth')
    # trained_model, Loss, Acc, f1 = TE_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TE_MNL, lambda_epochs=lambda_epochs,
    #                                             N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
    #                                             VERBOSE=VERBOSE, save_model=save_model, model_filename='te_mnl_model.pth')
    # trained_model, Loss, Acc, f1 = TEL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TEL_MNL, lambda_epochs=lambda_epochs,
    #                                             extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
    #                                             N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
    #                                             VERBOSE=VERBOSE, save_model=save_model, model_filename='te_mnl_model.pth')

    _, te_acc1, _, te_f1, _, _, _ = model_predict(noise_X1, Q_TEST, y_TEST, trained_model)
    # _, te_acc1, te_f1, _, _, _, _ = t_model_predict(noise_X1, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)
    test_acc1.append(te_acc1)
    test_f1.append(te_f1)
    _, te_acc2, _, te_f2, _, _, _ = model_predict(noise_X2, Q_TEST, y_TEST, trained_model)
    # _, te_acc2, te_f2, _, _, _, _ = t_model_predict(noise_X2, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)
    test_acc2.append(te_acc2)
    test_f2.append(te_f2)
    _, te_acc3, _, te_f3, _, _, _ = model_predict(noise_X3, Q_TEST, y_TEST, trained_model)
    # _, te_acc3, te_f3, _, _, _, _ = t_model_predict(noise_X3, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)
    test_acc3.append(te_acc3)
    test_f3.append(te_f3)
    _, te_acc4, _, te_f4, _, _, _ = model_predict(noise_X4, Q_TEST, y_TEST, trained_model)
    # _, te_acc4, te_f4, _, _, _, _ = t_model_predict(noise_X4, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)
    test_acc4.append(te_acc4)
    test_f4.append(te_f4)
    # _, te_acc5, _, te_f5, _, _, _ = model_predict(noise_X5, Q_TEST, y_TEST, trained_model)
    # _, te_acc5, te_f5, _, _, _, _ = t_model_predict(noise_X5, Q_TEST, y_TEST, trained_model, N_EPOCHS, lambda_epochs)
    # test_acc5.append(te_acc5)
    # test_f5.append(te_f5)
    # TE_MNL/TEL_MNL


re_acc, st_acc = np.mean(test_acc1), np.std(test_acc1)
re_f1, st_f1 = np.mean(test_f1), np.std(test_f1)
print("######################################################")
print('test1 acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('test1 f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

re_acc, st_acc = np.mean(test_acc2), np.std(test_acc2)
re_f1, st_f1 = np.mean(test_f2), np.std(test_f2)
print("######################################################")
print('test2 acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('test2 f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

re_acc, st_acc = np.mean(test_acc3), np.std(test_acc3)
re_f1, st_f1 = np.mean(test_f3), np.std(test_f3)
print("######################################################")
print('test3 acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('test3 f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

re_acc, st_acc = np.mean(test_acc4), np.std(test_acc4)
re_f1, st_f1 = np.mean(test_f4), np.std(test_f4)
print("######################################################")
print('test4 acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
print('test4 f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))

# re_acc, st_acc = np.mean(test_acc5), np.std(test_acc5)
# re_f1, st_f1 = np.mean(test_f5), np.std(test_f5)
# print("######################################################")
# print('test5 acc: {:.3f}, std: {:.3f}'.format(re_acc, st_acc))
# print('test5 f1: {:.3f}, std: {:.3f}'.format(re_f1, st_f1))






