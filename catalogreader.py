import os
import ROOT
import numpy as np
import pandas as pd
from scipy import fftpack
from config import DATA_LATACD_FOLDER_PATH
from utils import from_met_to_datetime

class CatalogReader():
    """Class to read the catalog of runs and their properties"""
    
    def __init__(self, data_dir = None, from_lat = True, start = 0, end = -1):
        """
        Initialize the CatalogReader object.

        Parameters:
        - data_dir (str): The directory path where the catalog data is stored.
        - start (int): The index of the first run directory to consider.
        - end (int): The index of the last run directory to consider.
        """
        if from_lat:
            data_dir = DATA_LATACD_FOLDER_PATH
        self.data_dir = data_dir
        self.runs_roots = [f'{self.data_dir}/{filename}' for filename in os.listdir(data_dir)]
        self.runs_roots.sort()
        self.runs_roots = self.runs_roots[start:end]

        # self.h_names = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']
        # self.h_names = [f'rate_tile{i};1' for i in range(89)]
        self.h_names = ['hist_top', 'hist_Xpos', 'hist_Xneg', 'hist_Ypos', 'hist_Yneg']
        self.runs_times = {}
        self.runs_dict = {}

    def get_runs_roots(self):
        """
        Get the list of run directories.

        Returns:
        - runs_dirs (list): The list of run directories.
        """
        return self.runs_roots
    
    def get_runs_times(self):
        """
        Get the dictionary of run times.

        Returns:
        - runs_times (dict): The dictionary of run times.
        """
        return self.runs_times

    def get_runs_dict(self, runs_roots, binning = None):
        """
        Get the dictionary of runs and their properties.

        Parameters:
        - runs_dirs (list): The list of run directories to consider.
        - binning (int): The binning factor for the histograms (optional).
        - smooth (bool): Flag indicating whether to apply smoothing to the histograms (optional).

        Returns:
        - runs_dict (dict): The dictionary of runs and their properties.
        """
        for fname in runs_roots:
            froot = ROOT.TFile.Open(fname, 'read')
            hist = froot.Get(self.h_names[0])
            histx = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
            self.runs_dict[fname] = {'MET': histx}
            self.runs_dict[fname]['datetime'] = from_met_to_datetime(histx)
            self.runs_times[fname] = (self.runs_dict[fname]['datetime'][0], self.runs_dict[fname]['datetime'][-1])
            for h_name in self.h_names:
                hist = froot.Get(h_name)
                if binning:
                    hist.Rebin(binning)
                histc = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
                self.runs_dict[fname][h_name.split('hist_')[-1]] = histc
            froot.Close()
        return self.runs_dict

    def add_smoothing(self, tile_signal_df):
        """
        """
        histx = tile_signal_df['MET']
        time_step = histx[2] - histx[1]
        nyquist_freq = 0.5 / time_step
        freq_cut1 = 0.01 * nyquist_freq
        for h_name in tile_signal_df.keys():
            if h_name not in ('MET', 'datetime'):
                histc = tile_signal_df[h_name].to_list()
                sig_fft = fftpack.fft(histc)
                sample_freq = fftpack.fftfreq(len(histc), d=time_step)
                low_freq_fft1  = sig_fft.copy()
                low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
                filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
                tile_signal_df[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal_df
    
    def get_signal_df_from_catalog(self, runs_dict):
        """
        Get the signal dataframe from the catalog.

        Parameters:
        - runs_dict (dict): The dictionary of runs and their properties.

        Returns:
        - signal_dataframe (pd.DataFrame): The signal dataframe.
        """
        if len(runs_dict) > 1:
            catalog_df = pd.concat([pd.DataFrame(hist_dict) for hist_dict in runs_dict.values()], ignore_index=True)
        else:
            catalog_df = pd.DataFrame(list(runs_dict.values())[0])
        return catalog_df

if __name__ == '__main__':
    cr = CatalogReader(DATA_LATACD_FOLDER_PATH, 0, 2)
    runs_roots = cr.get_runs_roots()
    cr.get_runs_dict(runs_roots)