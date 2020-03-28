# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['MAD']

# Cell
import numpy as np

class MAD():
    '''
    classe responsavel por implemetar zscore robusto
    para detecção de anomalias.
    '''
    value = 0.6745 # Valor que para aproximas

    def __init__(self):
        pass

    def fit(self, x):
        self.mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
        self.median = np.nanmedian(x)

    def predict(self, x):
        ''' returns MAD(Median Absolute Deviation) for each point x'''

        return (0.6745*(x - self.median))/self.mad