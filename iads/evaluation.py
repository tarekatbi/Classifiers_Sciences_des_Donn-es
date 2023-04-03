# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import math

def crossval_strat(X, Y, n_iterations, iteration):
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    fold_size = n_samples // n_iterations

    start_idx = iteration * fold_size
    end_idx = (iteration + 1) * fold_size

    test_idx = idx[start_idx:end_idx]
    train_idx = np.concatenate((idx[:start_idx], idx[end_idx:]))

    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    return Xtrain, Ytrain, Xtest, Ytest


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne=sum(L)/len(L)
    
    ecart_type=0
    for pref in L:
        ecart_type=ecart_type+((pref-moyenne)*(pref-moyenne))
        
    return (moyenne,math.sqrt(ecart_type/len(L)))
    raise NotImplementedError("Vous devez implémenter cette fonction !")   
# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
