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
from matplotlib.path import Path
import matplotlib.pyplot as plt

try:
    from modules.config import SC_FOLDER_NAME
    from modules.utils import Time, Data, Logger, logger_decorator
    from modules.plotter import Plotter
except:
    from config import SC_FOLDER_NAME
    from plotter import Plotter
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
        self.raw_data: np.ndarray = None
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
        wget.download(url, out=dir_path, bar=None)

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

        cols_to_split = {
            name for name in self.raw_data.dtype.names if self.raw_data[name][0].size > 1}
        arr_list = [np.array(Time.from_met_to_datetime(self.raw_data['START']))]
        names = ['datetime']
        if excluded_columns is None:
            excluded_columns = []
        cols_to_add = {
            name for name in set(self.raw_data.dtype.names) - cols_to_split if name not in excluded_columns}
        
        arr_list.extend([self.raw_data[name] for name in cols_to_add])
        names.extend(cols_to_add)

        for name in cols_to_split:
            split_arr = self.raw_data[name]
            split_names = [f'{name}_{i}' for i in range(split_arr.shape[1])]
            arr_list.extend([arr.T[0] for arr in np.hsplit(split_arr, split_arr.shape[1])])
            names.extend(split_names)
            
        dataframe = np.rec.fromarrays(arrayList=arr_list, names=names)
        return dataframe

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

    @logger_decorator(logger)
    def saa_boundary(self):
        """The coordinates of the SAA boundary in latitude and East longitude
        
        Returns:
            (np.array, np.array): The latitude and East longitude values
        """
        lat_saa = np.array(
            [-28.000, -17.867, -7.733, 2.00, 6.500, 6.500, 5.00,
            1.000, -4.155, -5.880, -10.020, -17.404, -28.000, -28.000])
        lon_saa = np.array(
            [32.900, 16, -3.103, -22.605, -32.400, -40.400, -50.000,
            -65.000, -84.000, -93.200, -101.300, -98.300, -91.100, 31.900])
        return (lat_saa, lon_saa)

    @logger_decorator(logger)
    def add_saa_passage(self, sc_df: pd.DataFrame, prev_lon_lat) -> pd.DataFrame:
        """
        Adds the TIME_FROM_SAA column to the spacecraft data.

        Args:
            sc_df (pd.DataFrame): The spacecraft data.

        Returns:
            pd.DataFrame: The spacecraft data with the SAA column added.
        """
        lat_saa, lon_saa = self.saa_boundary()
        saa_path = Path(np.vstack((lon_saa, lat_saa)).T)
        sc_df_distance = np.sqrt((sc_df['LON_GEO'].diff())**2 + (sc_df['LAT_GEO'].diff())**2)
        sc_df_distance = (sc_df_distance > 1) & (sc_df_distance < 300)
        is_contained = saa_path.contains_points(np.array([sc_df['LON_GEO'], sc_df['LAT_GEO']]).T)
        sc_df['SAA_EXIT'] = is_contained & sc_df_distance
        if prev_lon_lat != 0:
            prev_dist = np.sqrt((sc_df.loc[0, 'LON_GEO'] - prev_lon_lat[0])**2 + (sc_df.loc[0, 'LAT_GEO'] - prev_lon_lat[1])**2)
            sc_df.loc[0, 'SAA_EXIT'] = is_contained[0] and prev_dist > 1 and prev_dist < 300
        time_from_saa = sc_df['SAA_EXIT'].to_numpy()
        was_in_saa = np.roll(time_from_saa, shift=1)
        time_from_saa = (time_from_saa & ~was_in_saa).astype(float)
        exit_time = sc_df.loc[time_from_saa == 1, 'START']
        exit_time[0] = sc_df['START'][0] if prev_lon_lat != 0 else 0 # to be checked
        exit_time = exit_time.reindex(sc_df.index).ffill()
        time_from_saa = sc_df['START'] - exit_time
        sc_df['TIME_FROM_SAA'] = time_from_saa
        return sc_df, (sc_df.iloc[- 1]['LON_GEO'], sc_df.iloc[- 1]['LAT_GEO'])

    @logger_decorator(logger)
    def add_sun_occultation(self, sc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the SUN_IS_OCCULTED column to the spacecraft data.

        Args:
            sc_df (pd.DataFrame): The spacecraft data.

        Returns:
            pd.DataFrame: The spacecraft data with the SUN_IS_OCCULTED column added.
        """

        ra_sun_rad = np.deg2rad(sc_df['RA_SUN'])
        dec_sun_rad = np.deg2rad(sc_df['DEC_SUN'])
        ra_zenith_rad = np.deg2rad(sc_df['RA_ZENITH'])
        dec_zenith_rad = np.deg2rad(sc_df['DEC_ZENITH'])

        cos_theta = (np.sin(dec_sun_rad) * np.sin(dec_zenith_rad) + np.cos(dec_sun_rad) * np.cos(dec_zenith_rad) * np.cos(ra_sun_rad - ra_zenith_rad))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.rad2deg(theta_rad)
        sc_df['SUN_IS_OCCULTED'] = (theta_deg > 90).astype(float)
        return sc_df

    @logger_decorator(logger)
    def get_good_quality_data(self, dataframe) -> pd.DataFrame:
        '''
        Returns the dataframe masked for good data quality.

        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data with good data quality.
        '''
        return dataframe[dataframe['DATA_QUAL'] != 0].reset_index(drop=True)

if __name__ == '__main__':
    import time
    prev_lon_lat = 0
    dfs = []
    for week in [811, 812, 813]:
        start = time.time()
        sco = SpacecraftOpener()
        file = sco.get_sc_lat_weekly(week)
        sco.open(file, excluded_columns=['IN_SAA'])
        sc_params_df = sco.get_dataframe()
        sc_params_df = sco.get_good_quality_data(sc_params_df)
        sc_params_df, prev_lon_lat = sco.add_saa_passage(sc_params_df, prev_lon_lat)
        sc_params_df = sco.add_sun_occultation(sc_params_df)
        print(time.time() - start)
        dfs.append(sc_params_df)
    catalog_df = pd.concat(dfs, ignore_index=True)
    Plotter(df = catalog_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in catalog_df.columns if col not in ['TIME_FROM_SAA', 'RA_SUN', 'RA_ZENITH']], marker = ',', show = True, smoothing_key='smooth')
