import torch
import numpy as np
from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST, X_vars, Q_vars, Q_df_TRAIN, cats2ints_mapping, NUM_CHOICES
from models_pytorch.models import E_MNL, EL_MNL
from models_pytorch.utils import E_MNL_train, EL_MNL_train, model_predict
from models_pytorch.estimation import model_summary, get_betas_and_embeddings
import matplotlib.pyplot as plt
from adjustText import adjust_text


extra_emb_dims = 2
n_nodes = 15

trained_e_mnl_model, Loss, Acc, LL = E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
                                                 N_EPOCHS=200, LR=0.001, l2=0, BATCH_SIZE=50, drop=0.2, VERBOSE=1,
                                                 save_model=0, model_filename='e_mnl_model.pth')
#
# print(model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_e_mnl_model))
# print(model_predict(X_TEST, Q_TEST, y_TEST, trained_e_mnl_model))

# trained_el_mnl_model, Loss2, Acc2, LL2 = EL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, EL_MNL, extra_emb_dims, n_nodes,
#                                                       N_EPOCHS=200, LR=0.001, l2=0, BATCH_SIZE=50, drop=0.2,
#                                                       VERBOSE=1, save_model=0, model_filename='el_mnl_model.pth')
#
# print(model_predict(X_TRAIN, Q_TRAIN, y_TRAIN, trained_el_mnl_model))
# print(model_predict(X_TEST, Q_TEST, y_TEST, trained_el_mnl_model))

# # 如果训练完毕，加载权重
# e_mnl_model_weights = torch.load('model_weights.pth')
# e_mnl_model = E_MNL(NUM_CONT_VARS, NUM_EMB_VARS, NUM_CHOICES, NUM_UNIQUE_CATS)
# e_mnl_model.load_state_dict(e_mnl_model_weights)

# stats_EMNL = model_summary(trained_e_mnl_model, X_TRAIN, Q_TRAIN, y_TRAIN, X_vars_names=X_vars, Q_vars_names=Q_vars)
# print(stats_EMNL)

# Extract betas end embedding values from the trained models
EMNL_dict = get_betas_and_embeddings(trained_e_mnl_model, Q_df_TRAIN)
# print(EMNL_dict.keys())
#
cats2embs = dict(zip(cats2ints_mapping.keys(), EMNL_dict['embeddings']))
x_vars_betas = dict(zip(X_vars, EMNL_dict['betas_exog']))
q_vars_betas = dict(zip(Q_vars, EMNL_dict['betas_embs']))
#
# print('Betas for X variables:', x_vars_betas)
# print('Betas for Q variables:', q_vars_betas)
# print('Embeddings for categories in Q: ', cats2embs)

# create a dictionary that maps the Q variables to their corresponding categories
vars_cats = {col: sorted(set(Q_df_TRAIN[col])) for col in Q_df_TRAIN.columns}
# Visualize Interpretable Embeddings Dimensions
# for var, cats in vars_cats.items():
#     beta = q_vars_betas[var].cpu().detach().numpy()
#     # color= colors[idx]
#     embs = []
#     for cat in cats:
#         embs.append(cats2embs[cat][:NUM_CHOICES].cpu().detach().numpy())
#     embs = np.array(embs)  # *beta
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 8))
#     fig.suptitle(var + ' (Beta: ' + str(np.round(beta, 2)) + ')', fontsize='x-large')
#     ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
#     ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
#     ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
#     ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
#     texts = [ax1.text(embs[i, 0], embs[i, 1], cats[i], fontsize='x-large') for i in range(len(cats))]
#     texts2 = [ax2.text(embs[i, 0], embs[i, 2], cats[i], fontsize='x-large') for i in range(len(cats))]
#     print()
#     ax1.scatter(embs.T[0], embs.T[1], c='r',
#                 label=var + ' (Beta: ' + str(np.round(beta, 2)) + ')')  # list_betas_embs[idx] )
#     ax2.scatter(embs.T[0], embs.T[2], c='r',
#                 label=var + ' (Beta: ' + str(np.round(beta, 2)) + ')')  # list_betas_embs[idx] )
#     ax1.set_xlabel(xlabel='TRAIN (dim-1)', fontsize='x-large')
#     ax1.set_ylabel(ylabel='SM (dim-2)', fontsize='x-large')
#     ax2.set_xlabel(xlabel='TRAIN (dim-1)', fontsize='x-large')
#     ax2.set_ylabel(ylabel='CAR (dim-3)', fontsize='x-large')
#     ax1.grid()
#     ax2.grid()
#     adjust_text(texts, ax=ax1)
#     adjust_text(texts2, ax=ax2)
#     plt.legend(fontsize='large', loc='best')
#     plt.show()


# cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# all_colors = []
# all_embs = []
# all_cats = []
# all_vars = []
# # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# c = 0
# for var, cats in vars_cats.items():
#     if var not in ['ORIGIN', 'DEST']:
#         color = cycle[c]
#         c += 1
#         all_vars.append(var)
#         beta = q_vars_betas[var].cpu().detach().numpy()
#         for cat in cats:
#             all_colors.append(color)
#             all_embs.append(cats2embs[cat][:NUM_CHOICES].cpu().detach().numpy() * beta)
#             all_cats.append(cat)
# all_embs = np.array(all_embs)
# ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
# ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
# ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
# ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
# ax1.scatter(all_embs.T[0], all_embs.T[1], c=all_colors)
# ax2.scatter(all_embs.T[0], all_embs.T[2], c=all_colors)
# # texts1 = [ax1.annotate(all_cats[i], (all_embs[i, 0], all_embs[i, 1]),  fontsize='x-large') for i in range(len(all_cats))]
# texts1 = [ax1.text(all_embs[i, 0], all_embs[i, 1], all_cats[i], fontsize='x-large') for i in range(len(all_cats))]
# texts2 = [ax2.text(all_embs[i, 0], all_embs[i, 2], all_cats[i], fontsize='x-large') for i in range(len(all_cats))]
# ax1.set_xlabel(xlabel='TRAIN (dim-1)\nBeta SCALED', fontsize='x-large')
# ax1.set_ylabel(ylabel='SM (dim-2)\nBeta SCALED', fontsize='x-large')
# ax2.set_xlabel(xlabel='TRAIN (dim-1)\nBeta SCALED', fontsize='x-large')
# ax2.set_ylabel(ylabel='CAR (dim-3)\nBeta SCALED', fontsize='x-large')
# ax1.set_xlim([min(all_embs[:, 0]) - 0.2, max(all_embs[:, 0]) + 0.2])
# ax1.set_ylim([min(all_embs[:, 1]) - 0.2, max(all_embs[:, 1]) + 0.2])
# ax2.set_xlim([min(all_embs[:, 0]) - 0.2, max(all_embs[:, 0]) + 0.2])
# ax2.set_ylim([min(all_embs[:, 2]) - 0.2, max(all_embs[:, 2]) + 0.2])
# ax1.grid()
# ax2.grid()
# # adjust_text(texts1, ax=ax1)
# # adjust_text(texts2, ax=ax2)
# plt.show()

