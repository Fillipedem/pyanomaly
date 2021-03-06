# AUTOGENERATED! DO NOT EDIT! File to edit: 01_timeseries.ipynb (unless otherwise specified).

__all__ = ['smad']

# Cell
import numpy as np
import pandas as pd

from .stats import MAD, Tukey
from .utils import plot_anomalies

# Cell
from statsmodels.tsa.seasonal import STL

def smad(ts, m=3.0, period=None, stl_seasonal=25,
         only_low_values=False, score=False):
    '''
        Seasonal-MAD

        Input:
            ts: pd.Series with DateTimeIndex
            m:  stardard deviation
            period: time series seasonal periodo
            stl_seasonal: STL Seasonal parameter
            only_low_values: return anomalies only for low values
            score: if True returns the decision function
        Output:
    '''
    # Seasonal component according to the Papper
    if period is not None:
        stl = STL(ts, period=period, seasonal=stl_seasonal)
    else:
        stl = STL(ts, seasonal=stl_seasonal)
    res = stl.fit() # fit
    # calculamos o residuo
    residuo = ts - np.nanmedian(ts) - res.seasonal
    # Search outlier with mad
    mad = MAD(only_low_values=only_low_values)
    mad.fit(residuo)
    # return
    if score:
        return mad.decision_function(residuo)
    else:
        index = mad.predict(residuo, m=m).index
        return ts.loc[index]