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

import wget
from astropy.io import fits
from astropy.table import Table, vstack
import pandas as pd
import numpy as np
from scripts.config import SC_LAT_WEEKLY_FILE_PATH, regenerate_lat_weekly
from scripts.utils import Time, Data, Logger, logger_decorator


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
    def get_from_lat_weekly(self):
        """
        Retrieves data from LAT weekly files and writes them to a single FITS file.

        Returns:
            None
        """
        sc_fits_list = []
        SC_FILE_PATHS_FROM_LAT = regenerate_lat_weekly()
        print(SC_FILE_PATHS_FROM_LAT)
        for SC_FILE_PATH_FROM_LAT in SC_FILE_PATHS_FROM_LAT:
            sc_fits_list.append(Table.read(SC_FILE_PATH_FROM_LAT))
        vstack(sc_fits_list, join_type='outer', metadata_conflicts='warn').write(
            SC_LAT_WEEKLY_FILE_PATH, format='fits', overwrite=True)

    @logger_decorator(logger)
    def download_lat_weekly(self, start, end):
        """
        Downloads LAT weekly spacecraft data.

        Returns:
            None
        """
        # command = f'wget -m -P {SC_FOLDER_NAME_FROM_LAT_WEEKLY} -nH --cut-dirs=4 -np -e robots=off '
        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/'
        files_list = [
            url + f'lat_1sec_spacecraft_weekly_w{i}_p310_v001.fits' for i in range(start, end)]
        for file_url in files_list:
            wget.download(file_url, out='.')  # SC_FOLDER_NAME_FROM_LAT_WEEKLY)
        # command = command + ' '.join(files_list)
        # subprocess.run(command, shell=True, check=False)

    @logger_decorator(logger)
    def open(self, sc_filename=SC_LAT_WEEKLY_FILE_PATH, excluded_columns=None):
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
    # sc.download_lat_weekly(800, 802)
    sc.get_from_lat_weekly()
