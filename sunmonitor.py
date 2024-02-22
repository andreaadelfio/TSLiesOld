import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import time_support

from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

def find_goes_data(tstart, tend):
    """
    Find and download GOES XRS data for a specific time period.
    """
    result_goes = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("flx1s"))
    file_goes = Fido.fetch(result_goes, progress = False)
    goes_list = ts.TimeSeries(file_goes)
    df = pd.DataFrame([], columns=['datetime'])
    dfs = []
    for goes in goes_list:
        df_goes = pd.DataFrame(goes.to_dataframe())
        df_goes.index.name = 'datetime'
        df_goes.reset_index(inplace=True)
        df_goes = df_goes[(df_goes["xrsa_quality"] == 0) & (df_goes["xrsb_quality"] == 0)]
        df_goes = df_goes[['datetime', 'xrsb']]
        dfs.append(df_goes)
    xrsb_values = []

    for df in dfs:
        xrsb_values.append(df.set_index('datetime')['xrsb'])

    # Calcola la media di "xrsb" per ciascun "datetime"
    df_mean = pd.concat(xrsb_values, axis=1).mean(axis=1).reset_index()
    df_mean.columns = ['datetime', 'solar']
    return df_mean

if __name__ == "__main__":
    tstart, tend = '2023-12-05 09:35:00', '2023-12-05 12:44:25'
    find_goes_data(tstart, tend)