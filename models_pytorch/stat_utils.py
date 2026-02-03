import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def vuong_test(loglike1, loglike2, correction=True):
    # number of observations and check of models
    N = loglike1.shape[0]
    N2 = loglike2.shape[0]
    if N != N2:
        raise ValueError('Models do not have the same number of observations')
    # extract the log-likelihood for individual points with the models
    m = loglike1.cpu().detach().numpy() - loglike2.cpu().detach().numpy()
    # calculate the LR statistic
    LR = np.sum(m)
    # calculate the AIC and BIC correction factors -> these go to zero when df is same between models
    AICcor = 0
    BICcor = np.log(N)*AICcor/2
    # calculate the omega^2 term
    omega2 = np.var(m, ddof=1)
    # calculate the Z statistic with and without corrections
    Zs = np.array([LR,LR-AICcor,LR-BICcor])
    Zs /= np.sqrt(N*omega2)
    # calculate the p-value
    ps = []
    msgs = []
    for Z in Zs:
        if Z>0:
            ps.append(1 - norm.cdf(Z))
            msgs.append('model 1 preferred over model 2')
        else:
            ps.append(norm.cdf(Z))
            msgs.append('model 2 preferred over model 1')
    # share information
    print('=== Vuong Test Results ===')
    labs = ['Uncorrected']
    if AICcor!=0:
        labs += ['AIC Corrected','BIC Corrected']
    for lab,msg,p,Z in zip(labs,msgs,ps,Zs):
        print('  -> '+lab)
        print('    -> '+msg)
        print('    -> Z: '+str(Z))
        print('    -> p: '+str(p))

