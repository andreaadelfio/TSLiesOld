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
import pandas as pd
import numpy as np
from config import SC_FILE_PATH

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
        self.data = None
        self.df = None

    def open(self, sc_filename = SC_FILE_PATH):
        """
        Opens the spacecraft data file and retrieves the necessary information.

        Args:
            sc_filename (str): The path to the spacecraft data file. Defaults to SC_FILE_PATH.

        Returns:
            None
        """
        with fits.open(sc_filename) as hdulist:
            self.sc_tstart = hdulist[0].header['TSTART']
            self.sc_tstop = hdulist[0].header['TSTOP']
            self.data = hdulist[1].data

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

    def get_dataframe(self, data_to_df) -> pd.DataFrame:
        """
        Returns the dataframe containing the spacecraft data.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return pd.DataFrame({name: data_to_df.field(name).tolist() for name in data_to_df.names})

    def get_masked_dataframe(self, start, stop):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        mask = (self.data['START'] >= start) & (self.data['START'] <= stop)
        masked_data = self.data[mask]
        masked_data = {name: masked_data.field(name).tolist() for name in masked_data.names}
        return pd.DataFrame(masked_data)
    
    def get_masked_data(self, start, stop):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        mask = (self.data['START'] >= start) & (self.data['START'] <= stop)
        masked_data = self.data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

if __name__ == '__main__':
    sc = SpacecraftOpener()
    sc.open()
    print(sc.get_tstart())
    print(sc.get_tstop())
    #print(sc.get_data())
    #print(sc.get_dataframe(sc.get_data()))
    print(sc.get_masked_data(239557417, 239557500))
# with fits.open(SC_FILE_PATH) as hdulist:
#     sc_tstart = hdulist[0].header['TSTART']
#     sc_tstop = hdulist[0].header['TSTOP']
#     print(sc_tstart, sc_tstop)
#     data = hdulist[1].data
#     mask = (data['START'] >= 239557417) & (data['START'] <= 239557500)
#     data = data[mask]
#     data = {name: data.field(name).tolist() for name in data.names}
#     df = pd.DataFrame(data)
#     print(df)
    # for name in data.names:
    #     print(len(data.field(name)))
    # for name in data.columns:
    #     print(name)