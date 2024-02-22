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
from utils import from_met_to_datetime_str, from_met_to_datetime
from scipy.interpolate import interp1d

class SpacecraftOpener:
    '''
    A class for opening and retrieving information from a spacecraft data file
    '''
    def __init__(self):
        """
        Initializes a new instance of the SpacecraftOpener class.
        """
        self.sc_tstart = None
        self.sc_tstop = None
        self.sc_header = None
        self.data = None
        self.df: pd.DataFrame = None

    # def get_from_gbm(self, met_list):
    #     for met in met_list:
    #         ContinuousFtp(met=met[0]).get_poshist(SC_FOLDER_NAME_FROM_GBM)
    #     sc_fits_list = []
    #     SC_FILE_PATHS_FROM_GBM = regenerate_gbm()
    #     for SC_FILE_PATH_FROM_GBM in SC_FILE_PATHS_FROM_GBM:
    #         sc_fits_list.append(Table.read(SC_FILE_PATH_FROM_GBM))
    #     vstack(sc_fits_list, join_type='outer', metadata_conflicts='warn').write(SC_GBM_FILE_PATH, format='fits', overwrite=True)

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

    def open(self, sc_filename = SC_LAT_WEEKLY_FILE_PATH, from_gbm = False):
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
            self.sc_header = hdulist[0].header
            self.data = hdulist[1].data

    def get_sc_header(self):
        """
        Returns the spacecraft header.

        Returns:
            str: The spacecraft header.
        """
        return self.sc_header

    def get_data_columns(self) -> list:
        """
        Returns the names of the columns in the spacecraft data.
        
        Returns:
            list: The names of the columns in the spacecraft data.
        """
        return self.data.names

    def get_tstart(self) -> float:
        """
        Returns the start time of the spacecraft data.
        
        Returns:
            float: The start time of the spacecraft data.
        """
        return self.sc_tstart

    def get_tstop(self) -> float:
        """
        Returns the stop time of the spacecraft data.
        
        Returns:
            float: The stop time of the spacecraft data.
        """
        return self.sc_tstop
    
    def get_data(self, excluded_columns = []) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        cols_to_split = [name for name in self.data.dtype.names if self.data[name][0].size > 1]
        arr_list = [from_met_to_datetime(self.data['START'])]
        names = ['datetime']
        cols_to_add = [name for name in self.data.dtype.names if name not in excluded_columns]
        for name in cols_to_add:
            if name in cols_to_split:
                for i in range(self.data[name][0].size):
                    arr_list.append(self.data[name][:, i])
                    names.append(f'{name}_{i}')
            else:
                arr_list.append(self.data[name])
                names.append(name)
        new_data = np.rec.fromarrays(arrayList = arr_list, names = names)
        return new_data

    def convert_to_df(self, data_to_df) -> pd.DataFrame:
        """
        Converts the data containing the spacecraft data in a pd.DataFrame.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return pd.DataFrame({name: data_to_df.field(name).tolist() for name in data_to_df.dtype.names})

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe containing the spacecraft data.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return self.df

    def get_masked_dataframe(self, start, stop, data = None):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        if data is None:
            data = self.data
        mask = (data['datetime'] >= start) & (data['datetime'] <= stop)
        masked_data = data[mask]
        # masked_data = {name: masked_data.field(name).tolist() for name in masked_data.keys()}
        return pd.DataFrame(masked_data)
    
    def get_excluded_dataframes(self, data, start, stop):
        """
        Returns the excluded dataframes within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            list: The excluded dataframes within the specified time range.
        """
        mask = (data['datetime'] < start) | (data['datetime'] > stop)
        excluded_data = data[mask]
        return excluded_data

    def get_masked_data(self, start, stop):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        mask = (self.data['datetime'] >= start) & (self.data['START'] <= stop)
        masked_data = self.data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

    def filter_dataframe_with_run_times(self, initial_dataframe, run_times):
        """
        Returns the spacecraft dataframe filtered on runs times.
        
        Args:
            initial_dataframe (DataFrame): The initial spacecraft data.
            run_times (DataFrame): The dataframe containing run times.
        
        Returns:
            DataFrame: The filtered spacecraft dataframe.
        """
        df = pd.DataFrame()
        for start, end in run_times.values():
            df = pd.concat([df, self.get_masked_dataframe(data = initial_dataframe, start = start, stop = end)], ignore_index = True)
        return df
    
    def merge_dfs(self, first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(first_dataframe, second_dataframe, on = 'datetime', how = 'outer')

    
if __name__ == '__main__':
    sc = SpacecraftOpener()