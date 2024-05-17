'''
This module provides a class for opening and retrieving information from a spacecraft data file.

Classes:
    SpacecraftOpener: A class for opening and retrieving information from a spacecraft data file.

Methods:
    __init__(): Initializes a new instance of the SpacecraftOpener class.
    open(sc_filename = SC_FILE_PATH): Opens the spacecraft data file and retrieves the necessary information.
    get_tstart(): Returns the start time of the spacecraft data.
    get_tstop(): Returns the stop time of the spacecraft data.
    get_dataframe(): Returns the dataframe containing the spacecraft data.
    get_masked_data(start, stop): Returns the masked data within the specified time range.
'''
import os
import wget
from astropy.io import fits
import pandas as pd
import numpy as np
try:
    from scripts.config import SC_FOLDER_NAME
    from scripts.utils import Time, Data, Logger, logger_decorator
except:
    from config import SC_FOLDER_NAME
    from utils import Time, Data, Logger, logger_decorator

class SpacecraftOpener:
    '''
    A class for opening and retrieving information from a spacecraft data file
    '''
    logger = Logger('SpacecraftOpener').get_logger()

    @logger_decorator(logger)
    def __init__(self):
        """
        Initializes a new instance of the SpacecraftOpener class.
        """
        self.raw_data: None
        self.data = None

    @logger_decorator(logger)
    def get_sc_lat_weekly(self, week) -> list[str]:
        """
        Retrieves data from LAT weekly file names or downloads them if not found.

        Returns:
            list[str]: The list of LAT weekly file names.
        """
        sc_weekly_content = os.listdir(SC_FOLDER_NAME)
        filename = f'lat_1sec_spacecraft_weekly_w{week}_p310_v001.fits'
        if filename not in sc_weekly_content:
            self.download_lat_weekly(week)
        filename = os.path.join(SC_FOLDER_NAME, filename)
        return filename


    @logger_decorator(logger)
    def download_lat_weekly(self, week, dir_path=SC_FOLDER_NAME):
        """
        Downloads LAT weekly spacecraft data.

        Returns:
            None
        """
        url = f'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/1s_spacecraft/lat_1sec_spacecraft_weekly_w{week}_p310_v001.fits'
        wget.download(url, out=dir_path)

    @logger_decorator(logger)
    def open(self, sc_filename, excluded_columns=None):
        """
        Opens the spacecraft data file and retrieves the necessary information.

        Args:
            sc_filename (str): The path to the spacecraft data file. Defaults to SC_FILE_PATH.

        Returns:
            None
        """
        with fits.open(sc_filename) as hdulist:
            self.raw_data = hdulist[1].data
            self.data = self.init_data(excluded_columns=excluded_columns)

    @logger_decorator(logger)
    def init_data(self, excluded_columns=None) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        cols_to_split = [
            name for name in self.raw_data.dtype.names if self.raw_data[name][0].size > 1]
        arr_list = [Time.from_met_to_datetime(self.raw_data['START'])]
        names = ['datetime']
        if excluded_columns is None:
            excluded_columns = []
        cols_to_add = [
            name for name in self.raw_data.dtype.names if name not in excluded_columns]
        for name in cols_to_add:
            if name in cols_to_split:
                for i in range(self.raw_data[name][0].size):
                    arr_list.append(self.raw_data[name][:, i])
                    names.append(f'{name}_{i}')
            else:
                arr_list.append(self.raw_data[name])
                names.append(name)
        new_data = np.rec.fromarrays(arrayList=arr_list, names=names)
        return new_data

    @logger_decorator(logger)
    def get_data(self) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        return self.data

    @logger_decorator(logger)
    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe containing the spacecraft data.

        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return Data.convert_to_df(self.data)


if __name__ == '__main__':
    sc = SpacecraftOpener()
    files = sc.get_sc_lat_weekly([800, 801, 802])
    for file in files:
        sc.open(file)
        print(sc.get_dataframe().head())
