# AUTOGENERATED! DO NOT EDIT! File to edit: 00_stats.ipynb (unless otherwise specified).

__all__ = ['MAD', 'Tukey']

# Cell
import numpy as np

# Cell
class MAD():
    '''
    Robust z score implementation.

    Robust z score = x − μ1/2 MAD × 1.4826
    '''
    def __init__(self, only_low_values=False):
        self.only_low_values= only_low_values
        self.median = None
        self.mad = None

    def __mad(self, x):
        ''' retorna o MAD(Median Absolute Deviation) para cada valor de **x** '''
        return (x - self.median)/self.mad

    def fit(self, x):
        ''' Calcula os parametros do Zscore Robusto(Median/MAD) para os valores de **x** '''
        self.mad = 1.4826*np.nanmedian(np.abs(x - np.nanmedian(x)))
        self.median = np.nanmedian(x)

    def predict(self, x, m=3.0):
        ''' retorna se os valores de **x** são outliers '''
        assert m > 0
        assert len(x) > 0

        # Calcular MAD
        mad = self.__mad(x)

        if self.only_low_values: # Retornando anomalias apenas para os valores menores que -m
            return x[mad < -m]
        else:                    # MAD padrão, valores de anomalias maiores que m ou menores que -m
            return x[np.abs(mad) > m]

    def decision_function(self, x):
        ''' retorna se os valores de mad para cada valor em **x**'''
        mad = self.__mad(x)

        return mad

    def fit_predict(self, x, m=3.0):
        ''' Calcula os parametros e retorno os valores
            de **x** que são outliers'''
        self.fit(x)
        return self.predict(x, m)

# Cell
class Tukey():
    '''
    classe responsavel por implemetar Tukey Method
    para detecção de anomalias.
    '''

    def __init__(self, only_low_values=False):
        self.only_low_values = only_low_values
        self.iqr = None
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def fit(self, x):
        ''' Calcula os parametros do Tukey(Q1,Q2,Q3) para os valores de **x** '''
        x = np.sort(x)
        n = len(x)//2

        # calculando os quartiles
        self.q1 = np.nanmedian(x[:n])
        self.q2 = np.nanmedian(x)
        self.q3 = np.nanmedian(x[n:])

        self.iqr = self.q3 - self.q1
        self.min = self.q1 - 1.5*self.iqr
        self.max = self.q3 + 1.5*self.iqr

    def predict(self, x):
        ''' retorna se os valores de **x** são outliers '''
        if self.only_low_values:
            return x[(x < self.min)]
        else:
            return x[(x < self.min) | (x >= self.max)]

    def decision_function(self, x):
        ''' retorna o score para os valores de **x** '''
        score = np.zeros(len(x))
        score[x < self.min] = np.abs(x[x < self.min] - self.min)
        score[x > self.max] = np.abs(x[x > self.max] - self.max)

        return np.log(score + 1)

    def fit_predict(self, x):
        ''' Calcula os parametros e retorno os valores
            de **x** que são outliers'''
        self.fit(x)
        return self.predict(x)