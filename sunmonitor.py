import pandas as pd
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from plotter import Plotter
import os
from config import SOLAR_FOLDER_NAME
from utils import File, Time, Logger, logger_decorator


class SunMonitor:
    logger = Logger('SunMonitor').get_logger()

    @logger_decorator(logger)
    def __init__(self, tstart, tend):
        self.tstart = tstart
        self.tend = tend

    @logger_decorator(logger)
    def fetch_goes_data(self):
        """
        """
        result_goes = Fido.search(a.Time(self.tstart, self.tend), a.Instrument("XRS"), a.Resolution("flx1s"))
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
    
    # @logger_decorator(logger)
    def find_goes_data(self, file_goes_list):
        """
        Find and download GOES XRS data for a specific time period.

        Parameters:
        tstart (str): Start time of the data retrieval period in the format 'YYYY-MM-DD HH:MM:SS'.
        tend (str): End time of the data retrieval period in the format 'YYYY-MM-DD HH:MM:SS'.

        Returns:
        pandas.DataFrame: A DataFrame containing the mean solar XRSB data for each datetime.
        """
        dfs = []
        for file_goes in file_goes_list:
            print(file_goes)
            goes = ts.TimeSeries(file_goes)
            df_goes = pd.DataFrame(goes.to_dataframe())
            df_goes.index.name = 'datetime'
            df_goes.reset_index(inplace=True)
            df_goes = df_goes[(df_goes["xrsa_quality"] == 0) & (df_goes["xrsb_quality"] == 0)]
            df_goes = df_goes[['datetime', 'xrsb']]
            df_goes['datetime'] = Time.remove_milliseconds_from_datetime(df_goes['datetime'])
            dfs.append(df_goes)
        df_mean = pd.concat(dfs).groupby('datetime')['xrsb'].mean().reset_index()
        df_mean.columns = ['datetime', 'SOLAR']
        return df_mean


if __name__ == "__main__":
    sm = SunMonitor(tstart = '2024-02-16 00:00:00', tend = '2024-02-17 00:00:00')
    file_goes = sm.fetch_goes_data()
    df = sm.find_goes_data(file_goes)
    Plotter(df = df, label = 'solar activity').df_plot_tiles(x_col = 'datetime', excluded_cols=[], marker = ',')
    File.write_df_on_file(df, SOLAR_FOLDER_NAME + '/solar_activity')
    df = File.read_df_from_file(SOLAR_FOLDER_NAME + '/solar_activity')
    print(df.dtypes)