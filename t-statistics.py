from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST
from models_pytorch.models import E_MNL, EL_MNL, L_MNL, TE_MNL, TEL_MNL
from models_pytorch.trainer import E_MNL_train, EL_MNL_train, L_MNL_train, TE_MNL_train, TEL_MNL_train
from models_pytorch.utils import model_predict, t_model_predict, get_log_likelihood2, get_clarke, create_dataset, DataLoader
from models_pytorch.stat_utils import vuong_test
import torch
import torch.nn.functional as F


N_EPOCHS = 200 # 200
LR = 0.001
l2 = 1e-5
BATCH_SIZE = 50
drop = 0.2  # 0
VERBOSE = 0
save_model = 0

extra_emb_dims = 2
n_nodes = 15  # 15

lambda_epochs = 100

trained_model, Loss, Acc, LL, f1, Loss1 = E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
                                                      N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=1,
                                                      VERBOSE=VERBOSE, save_model=save_model,
                                                      model_filename='e_mnl_model.pth')
# _, _, _, _, te_loglike1, clarck1 = model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model)
#
trained_model, Loss, Acc, LL, f1, Loss2 = L_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, L_MNL,
                                               n_nodes=100, N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE,
                                               VERBOSE=VERBOSE, save_model=save_model,
                                               model_filename='l_mnl_model.pth')
# _, _, _, _, te_loglike2, clarck2 = model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model)


trained_model, Loss, Acc, LL, f1, Loss3 = E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
                                               N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
                                               VERBOSE=VERBOSE, save_model=save_model,
                                               model_filename='e_mnl_model.pth')
# _, _, _, _, te_loglike3, clarck3 = model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model)

trained_model, Loss, Acc, LL, f1, Loss4 = EL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, EL_MNL,
                                                extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
                                                N_EPOCHS=N_EPOCHS, LR=LR, l2=l2, BATCH_SIZE=BATCH_SIZE, drop=drop,
                                                VERBOSE=VERBOSE, save_model=save_model,
                                                model_filename='el_mnl_model.pth')
# _, _, _, _, te_loglike4, clarck4 = model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model)

# trained_model, Loss, Acc, f1 = TE_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TE_MNL, lambda_epochs=lambda_epochs,
#                                             N_EPOCHS=N_EPOCHS, LR=0.005, l2=l2, BATCH_SIZE=100, drop=0,
#                                             VERBOSE=VERBOSE, save_model=save_model,
#                                             model_filename='te_mnl_model.pth')
# _, _, _, te_loglike5, clarck5 = t_model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model, N_EPOCHS, lambda_epochs)
#
# trained_model, Loss, Acc, f1 = TEL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, TEL_MNL, lambda_epochs=lambda_epochs,
#                                              extra_emb_dims=extra_emb_dims, n_nodes=n_nodes,
#                                              N_EPOCHS=N_EPOCHS, LR=0.005, l2=l2, BATCH_SIZE=100, drop=0,
#                                              VERBOSE=VERBOSE, save_model=save_model,
#                                              model_filename='te_mnl_model.pth')
# _, _, _, te_loglike6, clarck6 = t_model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_model, N_EPOCHS, lambda_epochs)

torch.save(Loss1, 'loss1.pt')
torch.save(Loss2, 'loss2.pt')
torch.save(Loss3, 'loss3.pt')
torch.save(Loss4, 'loss4.pt')








# output6 = torch.load('output.pt')
# y6 = torch.load('y.pt')
# # output66 = F.softmax(output6, dim=1)
# yy6 = torch.nn.functional.one_hot(y6, num_classes=3)
# te_loglike6 = get_log_likelihood2(output6, yy6)
# clarck6 = get_clarke(output6, yy6)
#
# vuong_test(te_loglike1, te_loglike6)
# vuong_test(te_loglike2, te_loglike6)
# vuong_test(te_loglike3, te_loglike6)
# vuong_test(te_loglike4, te_loglike6)
#
# print(clarck1 - clarck6)
# print(clarck2 - clarck6)
# print(clarck3 - clarck6)
# print(clarck4 - clarck6)


