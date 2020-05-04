# AUTOGENERATED! DO NOT EDIT! File to edit: 00_stats.ipynb (unless otherwise specified).

__all__ = ['MAD', 'Tukey', 'window', 'window']

# Cell
import numpy as np

# Cell
class MAD():
    '''
    classe responsavel por implemetar zscore robusto
    para detecção de anomalias.
    '''
    def __init__(self):
        pass

    def _mad(self, x):
        ''' retorna o MAD(Median Absolute Deviation) para cada valor de **x** '''
        return (0.6745*(x - self.median))/self.mad

    def fit(self, x):
        ''' Calcula os parametros do Zscore Robusto(Median/MAD) para os valores de **x** '''
        self.mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
        self.median = np.nanmedian(x)

    def predict(self, x, m=3.0):
        ''' retorna se os valores de **x** são outliers '''
        mad = self._mad(x)
        return x[np.abs(mad) > m]

    def decision_function(self, x):
        ''' retorna se os valores de mad para cada valor em **x**'''
        mad = self._mad(x)
        return np.abs(mad)

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

    def __init__(self):
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
        return x[(x < self.min) | (x >= self.max)]

    def fit_predict(self, x):
        ''' Calcula os parametros e retorno os valores
            de **x** que são outliers'''
        self.fit(x)
        return self.predict(x)

# Cell
def window(m, x):
    ''' calcula a probabilidade do intervalo ser um outlier
        com base nas medições individuais de cada medidor '''
    y_pred = m.predict(x)
    return y_pred.sum()/len(y_pred)

def window(m, x):
    ''' calcula a probabilidade do intervalo ser um outlier
        com base na soma das medições individuas '''
    y_pred = m.predict_proba(x)
    return y_pred.sum()