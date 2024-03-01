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

from astropy.io import fits
from astropy.table import Table, vstack
import pandas as pd
import numpy as np
from gbm.data import PosHist
from gbm import coords
from config import SC_GBM_FILE_PATH, SC_LAT_WEEKLY_FILE_PATH, regenerate_lat_weekly
from utils import Time, Data
from scipy.interpolate import interp1d

class SpacecraftOpener:
    '''
    A class for opening and retrieving information from a spacecraft data file
    '''
    def __init__(self):
        """
        Initializes a new instance of the SpacecraftOpener class.
        """
        self.raw_data: None
        self.data = None

    def fetch_sc_weekly_data(self, runs_times):
        """
        Calculates which week to download from the runs_times and downloads the weekly files from 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/1s_spacecraft/'.
        using wget.
        """
        week_list = self.get_week_from_run_times(runs_times)
        print(week_list)

    def get_week_from_run_times(self, runs_times):
        """
        Returns the week from the runs_times.

        Args:
            runs_times (dict): The runs_times dictionary.

        Returns:
            list: The list of weeks.
        """
        return Time.get_week_from_datetime([times[0] for times in runs_times.values()])

    def get_from_lat_weekly(self):
        """
        Retrieves data from LAT weekly files and writes them to a single FITS file.

        Returns:
            None
        """
        sc_fits_list = []
        SC_FILE_PATHS_FROM_LAT = regenerate_lat_weekly()
        for SC_FILE_PATH_FROM_LAT in SC_FILE_PATHS_FROM_LAT:
            sc_fits_list.append(Table.read(SC_FILE_PATH_FROM_LAT))
        vstack(sc_fits_list, join_type='outer', metadata_conflicts='warn').write(SC_LAT_WEEKLY_FILE_PATH, format='fits', overwrite=True)

    def open(self, sc_filename = SC_LAT_WEEKLY_FILE_PATH, excluded_columns = [], from_gbm = False):
        """
        Opens the spacecraft data file and retrieves the necessary information.

        Args:
            sc_filename (str): The path to the spacecraft data file. Defaults to SC_FILE_PATH.

        Returns:
            None
        """
        if from_gbm:
            sc_filename = SC_GBM_FILE_PATH
        with fits.open(sc_filename) as hdulist:
            self.raw_data = hdulist[1].data
            self.data = self.init_data(excluded_columns = excluded_columns)
    
    def init_data(self, excluded_columns = []) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        cols_to_split = [name for name in self.raw_data.dtype.names if self.raw_data[name][0].size > 1]
        arr_list = [Time.from_met_to_datetime(self.raw_data['START'])]
        names = ['datetime']
        cols_to_add = [name for name in self.raw_data.dtype.names if name not in excluded_columns]
        for name in cols_to_add:
            if name in cols_to_split:
                for i in range(self.raw_data[name][0].size):
                    arr_list.append(self.raw_data[name][:, i])
                    names.append(f'{name}_{i}')
            else:
                arr_list.append(self.raw_data[name])
                names.append(name)
        new_data = np.rec.fromarrays(arrayList = arr_list, names = names)
        return new_data
    
    def get_data(self) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        return self.data

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe containing the spacecraft data.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return Data.convert_to_df(self.data)

    
if __name__ == '__main__':
    sc = SpacecraftOpener()
    sc.fetch_sc_weekly_data(runs_times = {'run1': ['2024-02-16 00:00:00', '2024-02-28 00:00:00']})