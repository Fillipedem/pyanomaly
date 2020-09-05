# AUTOGENERATED! DO NOT EDIT! File to edit: 02_forecast.ipynb (unless otherwise specified).

__all__ = ['anomaly_arima', 'check_int', 'anomaly_holtwinters']

# Cell
import numpy as np
import pandas as pd

# pyanomaly
from .stats import MAD, Tukey
from .utils import plot_anomalies

# Auto-Arima
import pmdarima as pm
# Holt-Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Prophet
from fbprophet import Prophet

# Cell
def anomaly_arima(ts, train_split=0.4, d=None,
                       seasonal=False, seasonal_periods=7,
                       only_lower=False):
    '''
    Utiliza Auro-Arima para detectar anomalias.

    Input:
        Serie Temporal
        ts: Serie Temporal pd.Series() com DateTimeIndex

        Parametros de Treino
        train_split: % Porcentagem do conjunto de dados inicial usado para treinar o modelo.

        Parametros do ARIMA
        d: Número minimo de diferenciações
        seasonal: Se True o ARIMA irá modelar a sazonalidade
        seasonal_periods: Inteiro indicando o periodo da serie temporal

    Output:
        pd.Series com os valores de anomalia
    '''
    # checks
    if len(ts) == 0:
        raise ValueError('Time Series is Empty')

    if not isinstance(ts, pd.core.series.Series):
        raise ValueError('ts is not a pd.Series')

    if not isinstance(ts.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise ValueError('ts.index is not a DatetimeIndex')

    if train_split < 0 or train_split > 1:
        raise ValueError('train_split out of range, should be [0, 1]')

    # Initial Train Set
    start = ts.index[int(train_split*len(ts.index))]

    # Auto-Arima
    model = pm.auto_arima(ts[:start], seasonal=seasonal, d=d,
                          m=seasonal_periods, suppress_warnings=True)

    y_pred = []
    conf_int = []
    for y in ts[start:]:
        # predict
        y_p, y_int = model.predict(1, return_conf_int=True)
        # update
        model.update(y, suppress_warnings=True)
        # save values
        y_pred.append(y)
        conf_int.append(y_int)

    # search anomalies
    y_pred = np.array(y_pred)
    conf_int = np.array(conf_int).reshape(-1, 2)

    normal = check_int(y_pred, conf_int, only_lower)
    anomalies = ts[start:][~normal]

    return anomalies

# Cell
def check_int(y, conf_interval, only_lower=False):
    '''
    Check if y is inside the conf_interval
    Input:
        y: Numpy Array 1D
        conf_int: Numpy Array 2D with (-1, 2) Shape
                  [[9.0, 15.0],
                   [4.5,  8.7],
                   [..     ..],
                   [10.4,  13.2]]
    '''
    lower = y > conf_interval[:, 0]
    upper = y < conf_interval[:, 1]

    if only_lower:
        return lower
    else:
        return np.bitwise_and(lower, upper)

# Cell
def anomaly_holtwinters(ts, seasonal, seasonal_periods):
    '''
    Predict anomaly with one stepm ahead forecast.

    Input:
        # Serie Temporal
        ts: Serie Temporal pd.Series() com DateTimeIndex

        # Parametros do algoritmo
        seasonal: 'add', 'mul', None
        seasonal_periods: Inteiro indicando o periodo da serie temporal
    '''
    # Preprocessamento
    # HoltWinters com use_boxcox=True requer que todos os valores sejam positivos (X>0)
    min_val = ts.min()
    if min_val <= 0:
        ts = (ts - min_val) + 1

    # Train model
    model = ExponentialSmoothing(ts, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods)
    res = model.fit(use_boxcox=True)
    y_pred = res.fittedvalues

    # Procurando anomalias no residuo
    # Utilizando MAD com sigma 3(99.7% considerando a prob normal)
    mad = MAD()
    anomalias = mad.fit_predict(ts - y_pred)

    # Posprocessamento, desfazendo alterações nos valores da equação
    if min_val <= 0:
        ts = (ts + min_val) - 1

    return ts[anomalias.index]