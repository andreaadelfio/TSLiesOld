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
from astropy.io import fits
from astropy.table import Table, vstack
import pandas as pd
import numpy as np
from gbm.finder import ContinuousFtp
from gbm.data import PosHist
from gbm import coords
from config import SC_LAT_FILE_PATH, SC_GBM_FILE_PATH, SC_FOLDER_NAME_FROM_GBM, SC_LAT_WEEKLY_FILE_PATH, SC_FOLDER_NAME_FROM_LAT_WEEKLY, regenerate_lat_weekly, regenerate_gbm

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

    def get_from_gbm(self, met_list):
        for met in met_list:
            ContinuousFtp(met=met[0]).get_poshist(SC_FOLDER_NAME_FROM_GBM)
        sc_fits_list = []
        SC_FILE_PATHS_FROM_GBM = regenerate_gbm()
        for SC_FILE_PATH_FROM_GBM in SC_FILE_PATHS_FROM_GBM:
            sc_fits_list.append(Table.read(SC_FILE_PATH_FROM_GBM))
        vstack(sc_fits_list, join_type='outer', metadata_conflicts='warn').write(SC_GBM_FILE_PATH, format='fits', overwrite=True)

    def get_from_lat_weekly(self):
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
            # self.sc_tstart = hdulist[0].header['TSTART']
            # self.sc_tstop = hdulist[0].header['TSTOP']
            self.data = hdulist[1].data

    def get_sc_header(self):
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
    
    def get_data(self) -> np.ndarray:
        """
        Returns the spacecraft data.

        Returns:
            numpy.ndarray: The spacecraft data.
        """
        return self.data

    def convert_to_df(self, data_to_df) -> pd.DataFrame:
        """
        Converts the data containing the spacecraft data in a pd.DataFrame.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return pd.DataFrame({name: data_to_df.field(name).tolist() for name in data_to_df.names})

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
        if 'START' in data.names:
            mask = (data['START'] >= start) & (data['START'] <= stop)
        else:
            mask = (data['SCLK_UTC'] >= start) & (data['SCLK_UTC'] <= stop)
        masked_data = data[mask]
        masked_data = {name: masked_data.field(name).tolist() for name in masked_data.names}
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
        mask = (data['START'] < start) | (data['START'] > stop)
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
        if 'START' in self.data.names:
            mask = (self.data['START'] >= start) & (self.data['START'] <= stop)
        else:
            mask = (self.data['SCLK_UTC'] >= start) & (self.data['SCLK_UTC'] <= stop)
        masked_data = self.data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

    def get_sc_input_dataframe(self, initial_data, event_times):
        """
        Returns the spacecraft dataframe filtered on events times.
        
        Args:
            initial_data (DataFrame): The initial spacecraft data.
            event_times (DataFrame): The dataframe containing event times.
        
        Returns:
            DataFrame: The filtered spacecraft dataframe.
        """
        df = pd.DataFrame()
        for start, end in event_times.values():
            df = pd.concat([df, self.get_masked_dataframe(data = initial_data, start = start, stop = end)], ignore_index = True)
        return df
    
    def get_from_lat_weekly_poshist(self, dic_data):
        p_tmp = PosHist.open_from_lat(SC_LAT_WEEKLY_FILE_PATH)
        met_ts = dic_data['time']
        # time_filter = (met_ts >= p_tmp._times.min()) & (met_ts <= p_tmp._times.max())
        # for key in dic_data.keys():
        #     dic_data[key] = dic_data[key][time_filter]
        # met_ts = dic_data['time']
        # # # Add feature columns
        # TODO average the position over 4 seconds
        # Position and rotation
        var_tmp = p_tmp.get_eic(met_ts)
        dic_data['pos_x'] = var_tmp[0]
        dic_data['pos_y'] = var_tmp[1]
        dic_data['pos_z'] = var_tmp[2]
        var_tmp = p_tmp.get_quaternions(met_ts)
        dic_data['a'] = var_tmp[0]
        dic_data['b'] = var_tmp[1]
        dic_data['c'] = var_tmp[2]
        dic_data['d'] = var_tmp[3]
        dic_data['lat'] = p_tmp.get_latitude(met_ts)
        dic_data['lon'] = p_tmp.get_longitude(met_ts)
        dic_data['alt'] = p_tmp.get_altitude(met_ts)
        # Velocity
        var_tmp = p_tmp.get_velocity(met_ts)
        dic_data['vx'] = var_tmp[0]
        dic_data['vy'] = var_tmp[1]
        dic_data['vz'] = var_tmp[2]
        # var_tmp = p_tmp.get_angular_velocity(met_ts)
        # dic_data['w1'] = var_tmp[0]
        # dic_data['w2'] = var_tmp[1]
        # dic_data['w3'] = var_tmp[2]
        # Sun and Earth visibility
        # dic_data['sun_vis'] = p_tmp.get_sun_visibility(met_ts)
        var_tmp = coords.get_sun_loc(met_ts)
        dic_data['sun_ra'] = var_tmp[0]
        dic_data['sun_dec'] = var_tmp[1]
        dic_data['earth_r'] = p_tmp.get_earth_radius(met_ts)
        var_tmp = p_tmp.get_geocenter_radec(met_ts)
        dic_data['earth_ra'] = var_tmp[0]
        dic_data['earth_dec'] = var_tmp[1]
        # Detectors pointing and visibility
        # for det_name in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']:
        #     # Equatorial pointing for each detector
        #     var_tmp = p_tmp.detector_pointing(det_name, met_ts)
        #     dic_data[det_name + '_' + 'ra'] = var_tmp[0]
        #     dic_data[det_name + '_' + 'dec'] = var_tmp[1]
        #     # Obscured by earth
        #     dic_data[det_name + '_' + 'vis'] = p_tmp.location_visible(var_tmp[0], var_tmp[1], met_ts)
        # Magnetic field
        # dic_data['saa'] = p_tmp.get_saa_passage(met_ts)
        dic_data['l'] = p_tmp.get_mcilwain_l(met_ts)
        # # # End add columns
        # Remove file if all the data are saved in dic_data
        # os.remove(PATH_TO_SAVE + FOLD_CSPEC_POS + '/' + file_tmp)
        return dic_data
    
if __name__ == '__main__':
    sc = SpacecraftOpener()
    sc.open(from_gbm = True)
    # print(sc.get_masked_data(239557417, 239557500))