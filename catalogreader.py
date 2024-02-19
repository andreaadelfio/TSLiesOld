import os
import argparse
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
from config import DATA_BAT_FOLDER_PATH, DATA_LATACD_FOLDER_PATH

class CatalogReader():
    """Class to read the catalog of runs and their properties"""
    
    def __init__(self, data_dir = None, from_lat = False, start = 0, end = -1):
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

        self.h_names_norm = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']
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

    def get_runs_dict(self, runs_roots, binning = None, smooth = False):
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
            # fname = f'{self.data_dir}/{run_dir}/ACDrates_{run_dir}_bin0.1.root'
            froot = ROOT.TFile.Open(fname, 'read')
            hist = froot.Get(self.h_names[0])
            histx = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
            self.runs_times[fname] = (histx[0], histx[-1])
            self.runs_dict[fname] = {}
            for h_name in self.h_names:
                hist = froot.Get(h_name)
                if binning:
                    hist.Rebin(binning)
                histc = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
                self.runs_dict[fname][h_name] = (histx, histc)
                if smooth:
                    freq_cut1 = 0.001
                    time_step = histx[2] - histx[1]
                    sig_fft = fftpack.fft(histc)
                    sample_freq = fftpack.fftfreq(len(histc), d=time_step)
                    low_freq_fft1  = sig_fft.copy()
                    low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
                    filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
                    self.runs_dict[fname][h_name] = (histx, histc, filtered_sig1)
            froot.Close()
        return self.runs_dict
    
    def get_signal_df_from_catalog(self, runs_dict):
        """
        Get the signal dataframe from the catalog.

        Parameters:
        - runs_dict (dict): The dictionary of runs and their properties.

        Returns:
        - signal_dataframe (pd.DataFrame): The signal dataframe.
        """
        signal_dataframe = pd.DataFrame()
        for run, hist_dict in runs_dict.items():
            run_signal = {'time': self.runs_times[run][0] + hist_dict['hist_top'][0]}
            for tile, signal in hist_dict.items():
                run_signal[tile] = signal[1]
            run_dataframe = pd.DataFrame(run_signal)
            signal_dataframe = pd.concat([signal_dataframe, run_dataframe], ignore_index=True)
        return signal_dataframe


    def main(self):
        """
        The main function of the CatalogReader class.
        """

        for ddir in self.runs_dirs:
            fname = f'{self.data_dir}/{ddir}/ACDrates_{ddir}_bin0.1.root'
            print(fname)

            froot = ROOT.TFile.Open(fname, 'read')
            for h_name_norm in self.h_names_norm:
                hist = froot.Get(h_name_norm)

                # DOPO RIAGGIUNGERE
                hist.Rebin(1)

                histc = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
                histx = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
                print(histc, histx)
                plot_name = f'{self.data_dir}/{ddir}/{ddir}_{h_name_norm}.png'

                freq_cut1   = 0.001

                time_step = histx[2] - histx[1]
                # The FFT of the signal
                sig_fft = fftpack.fft(histc)

                # The corresponding frequencies
                sample_freq = fftpack.fftfreq(len(histc), d=time_step)
                low_freq_fft1  = sig_fft.copy()
                low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
                filtered_sig1  = fftpack.ifft(low_freq_fft1)

                _, ax = plt.subplots(4, 1, figsize=(10,8))

                ax[0].set_title(f'{ddir} {h_name_norm}')
                ax[0].plot(histx, histc, label='_sum')
                ax[0].plot(histx, filtered_sig1, color='orange', label='cut 1')
                ax[0].legend()

                histrm100 = np.convolve(histc, np.ones(10)/10, mode='valid')
                nbin_conv = int(len(histrm100) - 1)
                for _ in range(0,9):
                    histrm100 = np.append(histrm100, histrm100[nbin_conv])

                SIG_POINT=[]
                SIG_POINT_4=[]
                SSIG_POINT_4=[]

                SIG_POINT = histrm100 - filtered_sig1
                sigma_histrm100 = np.std(SIG_POINT)
                xmax = np.max(histx)
                xmin = np.min(histx)
                ax[1].plot(histx, histc, color='green', label='sum')
                ax[1].plot(histx, histrm100, color='blue', label='sum')
                ax[1].plot(histx, filtered_sig1, color='orange', label='cut 1')
                ax[1].legend()

                ax[2].plot(histx, SIG_POINT, color="blue")
                ax[2].plot(SSIG_POINT_4, SIG_POINT_4, color="red")
                ax[2].hlines(y=sigma_histrm100*5, xmin=xmin, xmax=xmax, colors="blue")
                filter_step_c = []
                filter_mean_c = []
                sigma_step_c = []
                step_bin_c = []
                step_mean_c = []
                step_min_c = []
                step_max_c = []
                n_steps_c = 5
                durata_c = (xmax - xmin)/n_steps_c
                for y in range(int(n_steps_c)):
                    step_min_c = np.append(step_min_c, xmin + durata_c * y)
                    step_max_c = np.append(step_max_c, xmin + durata_c * (y + 1))
                    filter_step_c = np.logical_and(histx < step_max_c[y], histx > step_min_c[y])
                    filter_mean_c = np.append(filter_mean_c, SIG_POINT[filter_step_c])

                    sigma_step_c = np.append(sigma_step_c, np.std(filter_mean_c[~np.isnan(filter_mean_c)]))
                    step_bin_c = np.append(step_bin_c, abs(step_max_c-step_min_c)/2.)
                    step_mean_c = np.append(step_mean_c,(step_min_c+abs(step_max_c-step_min_c)/2.))

                ax[3].plot(histx, SIG_POINT, color="blue")
                ax[3].hlines(y=sigma_step_c*5, xmin=step_min_c, xmax=step_max_c, colors="blue")
                
                plt.show()
                plt.close('all')
            froot.Close()


if __name__ == '__main__':
    cr = CatalogReader(DATA_LATACD_FOLDER_PATH, 0, 2)
    runs_roots = cr.get_runs_roots()
    cr.get_runs_dict(runs_roots)