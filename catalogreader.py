import os
import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from config import DATA_FOLDER_PATH

class CatalogReader():
    """Class to read the catalog of GRBs and their properties"""
    
    def __init__(self, data_dir = DATA_FOLDER_PATH, start = 0, end = -1):
        self.data_dir = data_dir
        self.grbs_dirs = os.listdir(data_dir)
        self.grbs_dirs.sort()
        self.grbs_dirs = self.grbs_dirs[start:end]

        self.h_names_norm = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']
        self.h_names = ['hist_top', 'hist_Xpos', 'hist_Xneg', 'hist_Ypos', 'hist_Yneg']
        self.grb_times = {}
        self.grb_dict = {}

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Apply FFT selection to ACD rates', add_help=True)
        parser.add_argument('--fft_plot', action='store_true', default=True, help='Plot FFT analysis result for selected histograms')
        parser.add_argument('--running_plot', action='store_true', default=True, help='Plot running mean  for selected histograms')
        args = parser.parse_args()

        return args

    def get_grbs_dirs(self):
        return self.grbs_dirs
    
    def get_grb_times(self):
        return self.grb_times

    def get_grb_dict(self, grbs_dirs, binning = None, smooth = False):
        for grb_dir in grbs_dirs:
            fname = f'{self.data_dir}/{grb_dir}/ACDrates_{grb_dir}_bin0.1.root'
            froot = ROOT.TFile.Open(fname, 'read')
            hist = froot.Get(self.h_names[0])
            histx = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
            self.grb_times[grb_dir] = (float(hist.GetXaxis().GetTitle().split('-')[-1]), histx[0], histx[-1]) # get the GRB time from the histogram title ????
            self.grb_dict[grb_dir] = {}
            for h_name in self.h_names:
                hist = froot.Get(h_name)
                if binning:
                    hist.Rebin(binning)
                histc = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
                self.grb_dict[grb_dir][h_name] = (histx, histc)
                if smooth:
                    freq_cut1 = 0.001
                    time_step = histx[2] - histx[1]
                    sig_fft = fftpack.fft(histc)
                    sample_freq = fftpack.fftfreq(len(histc), d=time_step)
                    low_freq_fft1  = sig_fft.copy()
                    low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
                    filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
                    self.grb_dict[grb_dir][h_name] = (histx, histc, filtered_sig1)
            froot.Close()
        return self.grb_dict
    
    def main(self):
        args = self.parse_arguments()

        if args.fft_plot or args.running_plot:
            for ddir in self.grbs_dirs:
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
    cr = CatalogReader(DATA_FOLDER_PATH, 0, 2)
    grb_dirs = cr.get_grbs_dirs()
    cr.get_grb_dict(grb_dirs)