import sys,os
import numpy as np
import pandas as pd
import scipy
from scipy import fftpack, signal
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

import argparse

import matplotlib.pyplot as plt

import time
import json


####################

vdir = '/Volumes/SGDISK1/ACD/nfs/farm/g/glast/g/CRE/test_sm/BATgrbs'


grbs_dirs = os.listdir( vdir )
#################


#hnames     = [ 'hist_triggers' ] #, 'hist_top',     'hist_Xneg',	      'hist_Ypos',	     'hist_Yneg' ]
hnamesNorm = [ 'histNorm_top',  'histNorm_Xpos', 'histNorm_Xneg',	  'histNorm_Ypos',	 'histNorm_Yneg' ]

hnames  = [ 'hist_top',     'hist_Xpos',  'hist_Xneg',	      'hist_Ypos',	     'hist_Yneg' ]



def parse_arguments():

    parser = argparse.ArgumentParser(description='Apply FFT selection to ACD rates with Simulated Signal', add_help=True)
    parser.add_argument('--fft_plot',   action='store_true',  default=False,  help='Plot FFT analysis result for selected histograms')
    parser.add_argument('--running_plot',   action='store_true',  default=False,  help='Plot running mean  for selected histograms')

    
    args = parser.parse_args()

    return args


def get_signal( time, rate,  a0, t90  ):

    '''
    Add GRB-like  signal to trigger rates
    time : array with time information
    rate : array with original rate measurements 
    a0   : maximum amplitude of the signal
    t90  : GRB like signal t90
    '''

    t0 = t90/3
    signal = np.zeros( len(time) )
    tmask0 = (time >0 )      & (time < t0/5 )
    tmask1 = (time >= t0/5 ) & ( time < 3*5 )

    signal[tmask0] = a0* np.exp(  5*time[tmask0] / t0) 
    signal[tmask1] = a0* np.exp(  -(time[tmask1]-t0/10) / t0) 
    
    return rate+signal

    
def main():

    args = parse_arguments()
    
    grb_dict = {}


    if args.fft_plot == True or args.running_plot == True :
        
        ndir = 0
        for ddir in grbs_dirs:

            ndir = ndir +1

            if ndir > 5:
                break

            if ddir[4] != '9':
                continue

            fname = '{:}/{:}/ACDrates_{:}_bin0.1.root'.format(vdir, ddir, ddir)

            froot = ROOT.TFile.Open(fname, 'read')

            for hh, hhn in zip(hnames, hnamesNorm ):
                hist  = froot.Get(hh)
                histn = froot.Get(hhn)
                #hist  = froot.Get(hhn)
                #
                hist.Rebin(10)
                ##histn.Rebin(10)

                histc = np.array( [  hist.GetBinContent(ii)  for ii in range(1, hist.GetNbinsX()+1) ] )
                histx = np.array( [  hist.GetBinCenter(ii)   for ii in range(1, hist.GetNbinsX()+1) ] )


                histc =  get_signal( histx, histc, 1.5*np.mean(histc), 10 ) 
                histnc = np.array( [  histn.GetBinContent(ii)  for ii in range(1, histn.GetNbinsX()+1) ] )


                if args.fft_plot == True :
                    
                    """
                    bat64ms_fft_Fk  = np.fft.fft( bat64ms_flux )  / bat64ms_flux.shape[1]
                    #print('FFT Fk Shape', bat64ms_fft_Fk.shape )
                    bat64ms_fft_PS = np.absolute(bat64ms_fft_Fk)**2
                    
                    ps1 = bat64ms_fft_PS
                    nf = int(len(ps1[0])/2)
                    fp = np.arange(0, nf )
                    """

                    ###
                    #freq_cut = [ 2, 1, 0.5]
                    freq_cut1   = 0.001
                    freq_cut2   = 0.002
                    freq_cut3   = 0.003
                    time_step = histx[2]-histx[1]

                    # The FFT of the signal
                    sig_fft = fftpack.fft(histc)

                    # And the power (sig_fft is of complex dtype)
                    power = np.abs(sig_fft)**2
                    
                    # The corresponding frequencies
                    sample_freq = fftpack.fftfreq(len(histc), d=time_step)
                    
                    low_freq_fft1  = sig_fft.copy()
                    low_freq_fft2  = sig_fft.copy()
                    low_freq_fft3  = sig_fft.copy()
                    
                    low_freq_fft1[ np.abs(sample_freq) > freq_cut1] = 0
                    low_freq_fft2[ np.abs(sample_freq) > freq_cut2] = 0
                    low_freq_fft3[ np.abs(sample_freq) > freq_cut3] = 0

                    filtered_sig1  = fftpack.ifft(low_freq_fft1)
                    filtered_sig2  = fftpack.ifft(low_freq_fft2)
                    filtered_sig3  = fftpack.ifft(low_freq_fft3)
                    
                    fig, ax = plt.subplots(3,1, figsize=(10,8) )

                    #plt.title('{:} {:}'.format(ddir, hh))
                    ax[0].set_title('{:} {:}'.format(ddir, hh))
                    
                    ax[0].plot(sample_freq, power)
                    #ax[0].plot(power)
                    ax[0].set_xscale('log')
                    ax[0].set_yscale('log')

                    ax[1].plot(histx, histc)
                    ax[1].plot(histx, filtered_sig1,  color='orange',  label='cut 1')
                    ax[1].plot(histx, filtered_sig2,  color='red',     label='cut 2')
                    #ax[1].plot(filtered_sig3, color='gold',    label='cut 3')
                    plt.legend()
                    #ax[1].title('{:} {:}'.format(ddir, hh))

                    high_freq_fft1  = sig_fft.copy()
                    high_freq_fft2  = sig_fft.copy()
                    high_freq_fft1[ np.abs(sample_freq) < freq_cut1] = 0
                    high_freq_fft2[ np.abs(sample_freq) < freq_cut2] = 0

                    dfiltered_sig1  = fftpack.ifft(high_freq_fft1)
                    dfiltered_sig2  = fftpack.ifft(high_freq_fft2)
                    

                    #ax[2].plot(dfiltered_sig1,  color='orange',  label='cut 1')
                    #ax[2].plot(dfiltered_sig2,  color='red',     label='cut 2')


                    sig_diff1 = histc - filtered_sig1
                    sig_diff1 = histc - filtered_sig1
                    
                    ax[2].plot(histx, sig_diff1,  color='orange',  label='cut 1')
                    #ax[2].plot(histnc,     color='lightgray',  label='histn')
                    #ax[2].plot(sig_diff1-histnc,  color='green',  label='hndiff')

                    plt.legend()
                    plt.savefig('esempio_segnale.png')
                    plt.show()
                    
                if args.running_plot == True :
                    histrm5   = np.convolve(histc, np.ones(5)/5,     mode='valid')   #uniform_filter1d(histc, size=5)
                    histrm10  = np.convolve(histc, np.ones(10)/10,   mode='valid')   #uniform_filter1d(histc, size=5)
                    histrm100 = np.convolve(histc, np.ones(100)/100, mode='valid')   #uniform_filter1d(histc, size=10)

                    
                    
                    fig, ax = plt.subplots(3,1, figsize=(10,8) )

                    #plt.title('{:} {:}'.format(ddir, hh))
                    ax[0].set_title('{:} {:}'.format(ddir, hh))
                    #ax[0].plot(histx, histc)
                    #ax[0].plot(histx, histrm5,   color='orange',  label='Running Mean 5')
                    #ax[0].plot(histx, histrm10,  color='red',     label='Running Mean 10')
                    #ax[0].plot(histx, histrm100, color='green',   label='Running Mean 100')
                    ax[0].plot(histc)
                    ax[0].plot(histrm5,   color='magenta',  label='Running Mean 5')
                    ax[0].plot(histrm10,  color='orange',   label='Running Mean 10')
                    ax[0].plot(histrm100, color='red',      label='Running Mean 100')

                    plt.legend()
                    
                    plt.show()

                    
            froot.Close()



            
    

if __name__ == '__main__':

    main()
