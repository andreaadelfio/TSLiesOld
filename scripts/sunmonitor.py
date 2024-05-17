'''This module handles the solar activity retrieval'''
import os
import pandas as pd
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
try:
    from scripts.utils import Time, Logger, logger_decorator
    from scripts.plotter import Plotter
    from scripts.config import SOLAR_FOLDER_NAME
except:
    from utils import Time, Logger, logger_decorator
    from plotter import Plotter
    from config import SOLAR_FOLDER_NAME


class SunMonitor:
    '''Class to retrieve goes data'''
    logger = Logger('SunMonitor').get_logger()

    @logger_decorator(logger)
    def __init__(self, tstart, tend):
        self.tstart = tstart
        self.tend = tend

    @logger_decorator(logger)
    def fetch_goes_data(self):
        '''Fetches data from the GOES server'''
        result_goes = Fido.search(a.Time(self.tstart, self.tend), a.Instrument('XRS'), a.Resolution('flx1s'))
        files_to_fetch = {}
        files_list = []
        for i, url in enumerate(list(result_goes[0]['url'])):
            filename = url.split('/')[-1]
            path = os.path.join(SOLAR_FOLDER_NAME, filename)
            if os.path.exists(path):
                files_list.append(path)
            else:
                files_to_fetch[i] = path
        if len(files_to_fetch) > 0:
            for i in files_to_fetch.keys():
                files_list += Fido.fetch(result_goes[0][i], path=SOLAR_FOLDER_NAME, progress=False)
        return files_list

    @logger_decorator(logger)
    def merge_goes_data(self, goes_list) -> pd.DataFrame:
        '''
        Find and download GOES XRS data for a specific time period.

        Parameters:
        goes_list (list): list of GOES files to be merged.

        Returns:
        pandas.DataFrame: A DataFrame containing the mean solar XRSB data for each datetime.
        '''
        dfs = []
        for goes in goes_list:
            goes = ts.TimeSeries(goes)
            df_goes = pd.DataFrame(goes.to_dataframe())
            df_goes.index.name = 'datetime'
            df_goes.reset_index(inplace=True)
            df_goes = df_goes[(df_goes['xrsa_quality'] == 0) & (df_goes['xrsb_quality'] == 0)]
            df_goes = df_goes[['datetime', 'xrsb']]
            df_goes['datetime'] = Time.remove_milliseconds_from_datetime(df_goes['datetime'])
            dfs.append(df_goes)
        df_mean = pd.concat(dfs).groupby('datetime')['xrsb'].mean().reset_index()
        df_mean.columns = ['datetime', 'SOLAR']
        return df_mean


if __name__ == '__main__':
    sm = SunMonitor(tstart = '2024-02-01 00:00:00', tend = '2024-02-08 00:00:00')
    file_goes = sm.fetch_goes_data()
    df = sm.merge_goes_data(file_goes)
    Plotter(df = df, label = 'solar activity').df_plot_tiles(x_col = 'datetime', excluded_cols=[], marker = ',')
