import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from adjustText import adjust_text

import models_and_utils.models
from models_and_utils.models import E_MNL, EL_MNL
import models_and_utils.models_utils
from models_and_utils.models_utils import cats2ints, cats2ints_transform, E_MNL_train, EL_MNL_train, model_load_and_predict, model_predict
import models_and_utils.post_estimation_stats
from models_and_utils.post_estimation_stats import create_index, get_betas_and_embeddings, get_inverse_Hessian, get_stds, model_summary

import matplotlib.pyplot as plt

# Load the training data:
with open('../data/X_TRAIN.pkl', 'rb') as fp:
    X_TRAIN = pickle.load(fp)

Q_df_TRAIN = pd.read_csv('../data/Q_train.csv')

with open('../data/y_TRAIN.pkl', 'rb') as fp:
    y_TRAIN = pickle.load(fp)

# Load the test data:
with open('../data/X_TEST.pkl', 'rb') as fp:
    X_TEST = pickle.load(fp)

Q_df_TEST = pd.read_csv('../data/Q_test.csv')

with open('../data/y_TEST.pkl', 'rb') as fp:
    y_TEST = pickle.load(fp)

# Continuous variables (X) (plus intercepts)
X_vars=[ 'ASC_Car',
         'ASC_SM',
         'TT_SCALED(/100)',
         'COST_SCALED(/100)',
         'Headway_Train_SM']

# Categorical variables (Q) to be encoded as embeddings
Q_vars= ['PURPOSE', 'FIRST', 'TICKET', 'WHO',
         'LUGGAGE', 'AGE', 'MALE', 'INCOME',
         'GA', 'ORIGIN', 'DEST', 'SM_SEATS']

NUM_OBS_TRAIN= X_TRAIN.shape[0]
NUM_OBS_TEST= X_TEST.shape[0]

NUM_CHOICES= X_TRAIN.shape[2]
NUM_X_VARS= X_TRAIN.shape[1]
NUM_Q_VARS= Q_df_TRAIN.shape[-1]

UNIQUE_CATS= sorted(list(set(Q_df_TRAIN.values.reshape(-1))))
NUM_UNIQUE_CATS= len(UNIQUE_CATS)

print('The number of alternatives (C) is:',NUM_CHOICES)
print('The number of continuous variables (|X|) is:', NUM_X_VARS)
print('The number of categorical variables (|Q|) to be encoded as embeddings is:', NUM_Q_VARS)
print('The number of unique categories in Q (Z) is:', NUM_UNIQUE_CATS)
print()
print('The number of observations in the training set is:', NUM_OBS_TRAIN)
print('The number of observations in the test set is:', NUM_OBS_TEST)

cats2ints_mapping= cats2ints(Q_df_TRAIN)
Q_TRAIN= cats2ints_transform(Q_df_TRAIN, cats2ints_mapping)
Q_TEST= cats2ints_transform(Q_df_TEST, cats2ints_mapping)

# trained_emnl= E_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, E_MNL,
#                           N_EPOCHS=500, model_filename='name_your_model')



trained_emnl= EL_MNL_train(X_TRAIN, Q_TRAIN, y_TRAIN, EL_MNL,
                            n_extra_emb_dims=2,
                            N_NODES=15, N_EPOCHS=500,
                            model_filename='name_your_model')

stats_EMNL= model_summary(trained_emnl, X_TRAIN, Q_TRAIN, y_TRAIN,
                          X_vars_names= X_vars, Q_vars_names= Q_vars)
print(stats_EMNL)