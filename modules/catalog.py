'''
This module contains the class to manage the acd dataset.
'''
import os
import ROOT
import numpy as np
import pandas as pd
import re
from scipy import fftpack
try:
    from modules.config import DATA_LATACD_FOLDER_PATH
    from modules.utils import Time, Logger, logger_decorator, File
except:
    from config import DATA_LATACD_FOLDER_PATH
    from utils import Time, Logger, logger_decorator, File


class CatalogReader():
    '''Class to read the catalog of runs and their properties'''
    logger = Logger('CatalogReader').get_logger()

    @logger_decorator(logger)
    def __init__(self, h_names, data_dir = DATA_LATACD_FOLDER_PATH, start = 0, end = -1):
        '''
        Initialize the CatalogReader object.

        Parameters:
        - data_dir (str): The directory path where the catalog data is stored.
        - start (int): The index of the first run directory to consider.
        - end (int): The index of the last run directory to consider.
        '''
        self.data_dir = data_dir
        self.runs_roots = [f'{self.data_dir}/{filename}' for filename in os.listdir(data_dir)]
        self.runs_roots.sort()
        self.runs_roots = self.runs_roots[start:end]
        self.h_names = h_names
        self.runs_times = {}
        self.runs_dict = {}

    @logger_decorator(logger)
    def get_runs_roots(self):
        '''
        Get the list of run directories.

        Returns:
        - runs_dirs (list): The list of run directories.
        '''
        return self.runs_roots

    @logger_decorator(logger)
    def get_runs_times(self):
        '''
        Get the dictionary of run times.

        Returns:
        - runs_times (dict): The dictionary of run times.
        '''
        return self.runs_times

    @logger_decorator(logger)
    def get_runs_dict_root(self, runs_roots = None, binning = None):
        '''
        Fills a dictionary with (run_id)->(signals) for each run.

        Parameters:
        - runs_dirs (list): The list of run directories to consider.
        - binning (int): The binning factor for the histograms (optional).
        - smooth (bool): Flag indicating whether to apply smoothing to the histograms (optional).

        Returns:
        - runs_dict (dict): The dictionary of runs and their properties.
        '''
        if runs_roots is None:
            runs_roots = self.runs_roots
        for fname in runs_roots:
            froot = ROOT.TFile.Open(fname, 'read') # pylint: disable=maybe-no-member
            hist = froot.Get(self.h_names[0])
            histx = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
            datetime = np.array(Time.from_met_to_datetime(histx - 1))
            names = ['datetime', 'MET']
            arr_list = [datetime, histx]
            
            self.runs_times[fname] = (datetime[0], datetime[-1])
            for h_name in self.h_names:
                hist = froot.Get(h_name)
                if binning:
                    hist.Rebin(binning)
                histc = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])

                names.extend([h_name.split('hist_')[-1]])
                arr_list.extend([histc])
            self.runs_dict[fname] = np.rec.fromarrays(arrayList=arr_list, names=names)
            froot.Close()
        return self.runs_dict

    @logger_decorator(logger)
    def get_runs_dict(self, binning = None) -> pd.DataFrame:
        '''
        Get the pandas.Dataframe containing the signals for each run.

        Parameters:
        - binning (int): The binning factor for the histograms (optional, deprecated).

        Returns:
        - runs_dict (pandas.Dataframe): The dataframe containing the signals for each run.
        '''
        catalog_df = File.read_dfs_from_pk_folder(folder_path=self.data_dir, custom_sorter=lambda x: int(re.search(r"\d+", os.path.basename(x)).group(0)))
        catalog_df['datetime'] = np.array(Time.from_met_to_datetime(catalog_df['MET'] - 1))
        self.runs_times['catalog'] = (catalog_df['datetime'][0], catalog_df['datetime'].iloc[-1])
        return catalog_df

    @logger_decorator(logger)
    def add_smoothing(self, tile_signal):
        '''This function adds the smoothed histograms to the signal dataframe.'''
        histx = tile_signal['MET']
        time_step = histx[2] - histx[1]
        nyquist_freq = 0.5 / time_step
        freq_cut1 = 0.01 * nyquist_freq
        for h_name in set(tile_signal.keys()) - {'MET', 'datetime'}:
            histc = tile_signal[h_name].to_list()
            sig_fft = fftpack.fft(histc)
            sample_freq = fftpack.fftfreq(len(histc), d=time_step)
            low_freq_fft1  = sig_fft.copy()
            low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
            filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
            tile_signal[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal

    @logger_decorator(logger)
    def get_signal_df_from_catalog_root(self, runs_dict = None, binning = None):
        '''
        Get the signal dataframe from the catalog.

        Parameters:
        - runs_dict (dict): The dictionary of runs and their properties.

        Returns:
        - signal_dataframe (pd.DataFrame): The signal dataframe.
        '''
        if runs_dict is None:
            runs_dict = self.get_runs_dict_root(binning=binning)
        if len(runs_dict) > 1:
            catalog_df = pd.concat([pd.DataFrame(hist_dict) for hist_dict in runs_dict.values()], ignore_index=True)
        else:
            catalog_df = pd.DataFrame(list(runs_dict.values())[0])
        
        catalog_df = catalog_df[catalog_df['Xpos'] != 0]
        return catalog_df

    @logger_decorator(logger)
    def get_signal_df_from_catalog(self, runs_dict = None, binning = None):
        '''
        Get the signal dataframe from the catalog.

        Parameters:
        - runs_dict (dict): The dictionary of runs and their properties.

        Returns:
        - signal_dataframe (pd.DataFrame): The signal dataframe.
        '''
        if runs_dict is None:
            catalog_df = self.get_runs_dict(binning=binning)
        
        catalog_df = catalog_df[catalog_df['Xpos'] != 0]
        return catalog_df

if __name__ == '__main__':
    cr = CatalogReader(h_names=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg'], data_dir='data/LAT_ACD/output runs dfs', start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_catalog()
    print(len(tile_signal_df))