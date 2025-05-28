'''
This module contains the class to manipulate the runs from Fermi.
'''
import os
import sys
import concurrent.futures
import re
import glob
import ROOT
from tqdm import tqdm
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.plotter import Plotter
from modules.config import DATA_LATACD_PROCESSED_FOLDER_NAME, DATA_LATACD_RAW_FOLDER_NAME
from modules.utils import Logger, logger_decorator, File

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ACDReconRates:
    logger = Logger('ACDReconRates').get_logger()

    dict_tileSize = {}

    @logger_decorator(logger)
    def get_time0_last(self, tree):

        # TODO: qua si assume che il tree sia ordinato (ossia la prima entry corrisponda al met minore e l'ultima a quella maggiore...)
        # bisognerebbe cercare max e min  indipendentemente dall'ordine...

        nEntries = tree.GetEntries()
        tree.GetEntry(0)
        time0 = tree.time

        tree.GetEntry(nEntries - 1)
        time_last = tree.time

        return time0, time_last


    @logger_decorator(logger)
    def fill_dictSizes(self, inFileName):
        f = open(inFileName, "r")

        for line in f:
            splitted_line = line.split()
            tile = int(splitted_line[0])
            area = float(splitted_line[1])
            self.dict_tileSize[tile] = area


    @logger_decorator(logger)
    def mediaSides(self, hist_dict, identityFunc):

        # top:
        hist_top = hist_dict[0].Clone()
        hist_top.SetTitle("top")
        hist_top.SetName("hist_top")

        hist_top.Reset()
        # hist_top.Sumw2()

        n = 0.
        for tileId in range(64, 88 + 1):
            hist_top.Add(hist_dict[tileId])
            n = n + 1.

        hist_top.Divide(identityFunc, n)

        # X+:
        hist_Xpos = hist_dict[0].Clone()
        hist_Xpos.SetTitle("Xpos")
        hist_Xpos.SetName("hist_Xpos")
        hist_Xpos.Reset()
        # hist_top.Sumw2()
        n = 0.
        for tileId in range(48, 63 + 1):
            hist_Xpos.Add(hist_dict[tileId])
            n = n+1.

        hist_Xpos.Divide(identityFunc, n)

        # Xneg
        # 32 and tileId <= 47:
        hist_Xneg = hist_dict[0].Clone()
        hist_Xneg.SetTitle("Xneg")
        hist_Xneg.SetName("hist_Xneg")
        hist_Xneg.Reset()
        # hist_top.Sumw2()
        n = 0.
        for tileId in range(32, 47+1):
            hist_Xneg.Add(hist_dict[tileId])
            n = n+1.

        hist_Xneg.Divide(identityFunc, n)

        # Y+:
        hist_Ypos = hist_dict[0].Clone()
        hist_Ypos.SetTitle("Ypos")
        hist_Ypos.SetName("hist_Ypos")
        hist_Ypos.Reset()
        # hist_top.Sumw2()
        n = 0.
        # 16 and tileId <=31
        for tileId in range(16, 31+1):
            hist_Ypos.Add(hist_dict[tileId])
            n = n+1.

        hist_Ypos.Divide(identityFunc, n)

        # Yneg
        hist_Yneg = hist_dict[0].Clone()
        hist_Yneg.SetTitle("Yneg")
        hist_Yneg.SetName("hist_Yneg")
        hist_Yneg.Reset()

        n = 0.
        # >= 0 and tileId<=15:
        for tileId in range(0, 15+1):
            hist_Yneg.Add(hist_dict[tileId])
            n = n+1.

        hist_Yneg.Divide(identityFunc, n)

        return hist_top, hist_Xpos, hist_Xneg, hist_Ypos, hist_Yneg


    @logger_decorator(logger)
    def createTChain(self, rootfiles, treeName, run_path, end = None):
        run = re.search(r"\d+", os.path.basename(run_path)).group(0)
        try:
            chain = ROOT.TChain(treeName) # pylint: disable=maybe-no-member
        except Exception as e:
            with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                error_file.write(f"Error {e} in {run}\n")
        for line in rootfiles:
            if not line.endswith('.root'):
                rootfiles.remove(line)
        rootfiles.sort(key=lambda x: int(re.search(r"\d+", os.path.splitext(x)[0].split(run)[-1]).group(0)))
        for run in rootfiles[:end]:
            try:
                chain.Add(os.path.join(run_path, run))
            except Exception as e:
                with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                    error_file.write(f"Error {e} in {run}\n")
        return chain

    @logger_decorator(logger)
    def create_root(self, binning, output_filename, INPUT_ROOTS_FOLDER):
        import time
        start = time.time()
        list_file = os.listdir(INPUT_ROOTS_FOLDER)
        list_file.sort()
        output_rootfile = ROOT.TFile(f'{output_filename}.root', 'recreate') # pylint: disable=maybe-no-member
        myTree = self.createTChain(list_file, 'myTree', INPUT_ROOTS_FOLDER)

        time0, time_last = self.get_time0_last(myTree)
        n_bins = int((time_last - time0) / binning)
        identityFunc = ROOT.TF1("identityFunc", "1", time0, time_last) # pylint: disable=maybe-no-member

        # fill hist di tutti i triggers... serve per avere i rates normalizzati (i.e. acd occupancy)
        hist_triggers = ROOT.TH1F( # pylint: disable=maybe-no-member
            "hist_triggers", "hist_triggers", n_bins, time0, time_last)
        myString = 'time >> hist_triggers'
        myTree.Draw(myString, "", "goff")
        hist_triggers.Sumw2()
        # divido per larghezza bin => rate in Hz
        hist_triggers.Divide(identityFunc, binning)

        hist_dict = {}
        # histNorm_dict = {}

        for tileID in range(0, 89):
            hist_name = 'rate_tile'+str(tileID)
            hist_dict[tileID] = ROOT.TH1F( # pylint: disable=maybe-no-member
                hist_name, hist_name, n_bins, time0, time_last)

            string = 'time >> '+hist_name
            # cut="acdE_acdtile["+str(tileID)+"] >0.04"  # cut E>0.04!!!!!!!!!!!!
            cut = "acdE_acdtile["+str(tileID)+"] >0"

            myTree.Draw(string, cut, "goff")
            hist_dict[tileID].Sumw2()
            hist_dict[tileID].Divide(
                identityFunc, binning*self.dict_tileSize[tileID])
            hist_dict[tileID].GetXaxis().SetTitle('met')

            # histNorm_dict[tileID] = hist_dict[tileID].Clone()
            # hist_nameNorm = 'NORMrate_tile'+str(tileID)
            # histNorm_dict[tileID].SetName(hist_nameNorm)
            # histNorm_dict[tileID].SetTitle(hist_nameNorm)
            # histNorm_dict[tileID].GetXaxis().SetTitle('met')

            # histNorm_dict[tileID].Divide(hist_triggers)

            output_rootfile.cd()
            # hist_dict[tileID].Write()
            # histNorm_dict[tileID].Write()
            # print(f'\rCompletion: {tileID/89*100:.2f}%', end='')

        hist_top, hist_Xpos, hist_Xneg, hist_Ypos, hist_Yneg = self.mediaSides(
            hist_dict, identityFunc)
        # histNorm_top, histNorm_Xpos, histNorm_Xneg, histNorm_Ypos, histNorm_Yneg = mediaSides(
            # histNorm_dict, identityFunc)

        # histNorm_top.SetNameTitle('histNorm_top', 'histNorm_top')
        # histNorm_Xpos.SetNameTitle('histNorm_Xpos', 'histNorm_Xpos')
        # histNorm_Xneg.SetNameTitle('histNorm_Xneg', 'histNorm_Xneg')
        # histNorm_Ypos.SetNameTitle('histNorm_Ypos', 'histNorm_Ypos')
        # histNorm_Yneg.SetNameTitle('histNorm_Yneg', 'histNorm_Yneg')

        hist_triggers.Write()

        hist_top.Write()
        hist_Xpos.Write()
        hist_Xneg.Write()
        hist_Ypos.Write()
        hist_Yneg.Write()

        # histNorm_top.Write()
        # histNorm_Xpos.Write()
        # histNorm_Xneg.Write()
        # histNorm_Ypos.Write()
        # histNorm_Yneg.Write()

        output_rootfile.Close()
        print(time.time() - start)

    def resample_events(self, df, bin_edges, resample_rate):
        resampled_indices = []
        for i in range(len(bin_edges) - 1):
            bin_events = df[(df['time'] >= bin_edges[i]) & (df['time'] < bin_edges[i + 1])]
            if len(bin_events) > 0:
                resample_prob = min(resample_rate / len(bin_events), 1.0)
                mask = np.random.rand(len(bin_events)) < resample_prob
                resampled_indices.extend(bin_events[mask].index)
        return df.loc[resampled_indices]

    def create_df_norm(self, binning, output_filename, INPUT_ROOTS_FOLDER):
        tiles_faces = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}
        list_file = os.listdir(INPUT_ROOTS_FOLDER)
        myTree = self.createTChain(list_file,'myTree', INPUT_ROOTS_FOLDER)
        try:
            dict_np = ROOT.RDataFrame(myTree).AsNumpy() # pylint: disable=maybe-no-member
            df = pd.DataFrame(dict_np)
            # df.to_csv('test.csv')
            # notnull = df['acdE_acdtile'].apply(any)
            # df = df[notnull]
            # if 'gltGemEngine' in df.columns:
            #     df = df[df['gltGemEngine'] != 3]

            df.drop(columns=['gltGemSummary', 'gltGemEngine'], inplace=True)
            time0, time_last = df['time'].min(), df['time'].max()
            n_bins = int((time_last - time0) / binning)
            bin_edges = np.linspace(time0, time_last, n_bins+1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            return_df = pd.DataFrame(bin_centers, columns=['MET'])
            
            # triggs = df['time'].values
            # counts, _ = np.histogram(triggs, bins=bin_edges)
            # from matplotlib import pyplot as plt
            # plt.plot(bin_centers, counts)
            # plt.show()
            # print(f"Number of events: {len(df)}")
            resample_rate = 200 * binning

            df = self.resample_events(df, bin_edges, resample_rate)
            # counts, _ = np.histogram(df['time'].values, bins=bin_edges)
            # plt.plot(bin_centers, counts)
            # plt.show()

            for face, tiles in tiles_faces.items():
                face_rates_low = np.zeros(n_bins)
                face_rates_middle = np.zeros(n_bins)
                face_rates_high = np.zeros(n_bins)
                for tile in range(tiles[0], tiles[-1]):
                    acdE_acdtile = np.array([i[tile] for i in df['acdE_acdtile'].values]).T
                    times_with_energy_low = df['time'][(acdE_acdtile > 0.) & (acdE_acdtile <= 0.01)].values
                    times_with_energy_middle = df['time'][(acdE_acdtile > 0.01) & (acdE_acdtile <= 0.5)].values
                    times_with_energy_high = df['time'][(acdE_acdtile > 0.5)].values
                    counts_low, _ = np.histogram(times_with_energy_low, bins=bin_edges)
                    counts_middle, _ = np.histogram(times_with_energy_middle, bins=bin_edges)
                    counts_high, _ = np.histogram(times_with_energy_high, bins=bin_edges)
                    norm = 1 / binning / self.dict_tileSize[tile]
                    face_rates_low += counts_low * norm
                    face_rates_middle += counts_middle * norm
                    face_rates_high += counts_high * norm
                    
                # counts, _ = np.histogram(face_counts, bins=bin_edges)
                # plt.plot(bin_centers, face_counts)
                # plt.show()
                return_df[f"{face}_low"] = face_rates_low / (tiles[-1] - tiles[0])
                return_df[f"{face}_middle"] = face_rates_middle / (tiles[-1] - tiles[0])
                return_df[f"{face}_high"] = face_rates_high / (tiles[-1] - tiles[0])
            
            mask = return_df.loc[:, return_df.columns != 'MET'].any(axis=1)

            if len(return_df[mask]) != len(return_df):
                with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'dimensions_checks.txt'), 'a') as file:
                    file.write(f"{len(return_df[mask])} {len(return_df)} {INPUT_ROOTS_FOLDER}\n")
            # Plotter(df = return_df, label = 'Tiles').df_plot_tiles(x_col = 'MET',
            #                                                                 excluded_cols = [],
            #                                                                 marker = ',',
            #                                                                 smoothing_key='smooth',
            #                                                                 show = True)
            # Plotter(df = return_df_low, label = 'Tiles low').df_plot_tiles(x_col = 'MET',
            #                                                                 excluded_cols = [],
            #                                                                 marker = ',',
            #                                                                 smoothing_key='smooth',
            #                                                                 show = True)
            # Plotter(df = return_df_high, label = 'Tiles high').df_plot_tiles(x_col = 'MET',
            #                                                                 excluded_cols = [],
            #                                                                 marker = ',',
            #                                                                 smoothing_key='smooth',
            #                                                                 show = True)
            # Plotter.show()
            File.write_df_on_file(return_df,
                                    filename=output_filename,
                                    fmt='pk')
        except Exception as e:
            with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                error_file.write(f"Error {e} in {INPUT_ROOTS_FOLDER}\n")

    def create_df(self, binning, output_filename, INPUT_ROOTS_FOLDER):
        '''Create a dataframe with the countrates for each tile and each face of the ACD, with no resampling or energy cut.

        Args:
            binning (int): The binning for the histogram.
            output_filename (str): The output filename for the dataframe.
            INPUT_ROOTS_FOLDER (str): The input folder containing the ROOT files.
        '''
        tiles_faces = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}
        list_file = os.listdir(INPUT_ROOTS_FOLDER)
        myTree = self.createTChain(list_file,'myTree', INPUT_ROOTS_FOLDER)
        try:
            dict_np = ROOT.RDataFrame(myTree).AsNumpy() # pylint: disable=maybe-no-member
            df = pd.DataFrame(dict_np)
            # df.to_csv('test.csv')
            notnull = df['acdE_acdtile'].apply(any)
            df = df[notnull]
            if 'gltGemEngine' in df.columns:
                df = df[df['gltGemEngine'] != 3]

            df.drop(columns=['gltGemSummary', 'gltGemEngine'], inplace=True)
            time0, time_last = df['time'].min(), df['time'].max()
            n_bins = int((time_last - time0) / binning)
            bin_edges = np.linspace(time0, time_last, n_bins+1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            return_df = pd.DataFrame(bin_centers, columns=['MET'])
            
            for face, tiles in tiles_faces.items():
                face_rates = np.zeros(n_bins)
                
                for tile in range(tiles[0], tiles[-1]):
                    acdE_acdtile = np.array([i[tile] for i in df['acdE_acdtile'].values]).T
                    times_with_energy = df['time'][acdE_acdtile > 0].values
                    
                    counts, _ = np.histogram(times_with_energy, bins=bin_edges)
                    norm = 1 / binning / self.dict_tileSize[tile]
                    face_rates += counts * norm

                return_df[face] = face_rates / (tiles[-1] - tiles[0])
            
            mask = return_df.loc[:, return_df.columns != 'MET'].any(axis=1)

            if len(return_df[mask]) != len(return_df):
                with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'dimensions_checks.txt'), 'a') as file:
                    file.write(f"{len(return_df[mask])} {len(return_df)} {INPUT_ROOTS_FOLDER}\n")
            File.write_df_on_file(return_df,
                                    filename=output_filename,
                                    fmt='pk')
        except Exception as e:
            with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                error_file.write(f"Error {e} in {INPUT_ROOTS_FOLDER}\n")

    def create_df_energy_flux(self, binning, output_filename, INPUT_ROOTS_FOLDER):
        tiles_faces = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}
        list_file = os.listdir(INPUT_ROOTS_FOLDER)
        myTree = self.createTChain(list_file,'myTree', INPUT_ROOTS_FOLDER)
        try:
            dict_np = ROOT.RDataFrame(myTree).AsNumpy() # pylint: disable=maybe-no-member
            df = pd.DataFrame(dict_np)
            # df.to_csv('test.csv')
            # notnull = df['acdE_acdtile'].apply(any)
            # df = df[notnull]
            # if 'gltGemEngine' in df.columns:
            #     df = df[df['gltGemEngine'] != 3]

            df.drop(columns=['gltGemSummary', 'gltGemEngine'], inplace=True)
            time0, time_last = df['time'].min(), df['time'].max()
            n_bins = int((time_last - time0) / binning)
            bin_edges = np.linspace(time0, time_last, n_bins+1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            return_df = pd.DataFrame(bin_centers, columns=['MET'])

            resample_rate = 200
            df = self.resample_events(df, bin_edges, resample_rate)
            
            for face, tiles in tiles_faces.items():
                face_rates = np.zeros(n_bins)
                face_energies = np.zeros(n_bins)
                
                for tile in range(tiles[0], tiles[-1]):
                    acdE_acdtile = np.array([i[tile] for i in df['acdE_acdtile'].values]).T
                    # times_with_energy = df['time'][acdE_acdtile < 0.01].values
                    # energies = acdE_acdtile[acdE_acdtile < 0.01]
                    times_with_energy = df['time'][acdE_acdtile > 0].values
                    energies = acdE_acdtile[acdE_acdtile > 0]
                    # times_with_energy = df['time'][(acdE_acdtile > 0.01) & (acdE_acdtile < 0.5)].values
                    # energies = acdE_acdtile[(acdE_acdtile > 0.01) & (acdE_acdtile < 0.5)]
                    counts, _ = np.histogram(times_with_energy, bins=bin_edges)
                    energies_counts, _ = np.histogram(times_with_energy, bins=bin_edges, weights=energies)
                    norm = 1 / binning / self.dict_tileSize[tile]
                    face_rates += counts * norm
                    face_energies += energies_counts * norm

                return_df[face] = face_rates / (tiles[-1] - tiles[0])
                return_df[f'{face}_en'] = face_energies / (tiles[-1] - tiles[0])
            import matplotlib.pyplot as plt
            mask = return_df.loc[:, return_df.columns != 'MET'].any(axis=1)
            for face in tiles_faces.keys():
                plt.plot(return_df[f'{face}_en'], return_df[face], '.', label=face)
            plt.legend()
            plt.show()
            if len(return_df[mask]) != len(return_df):
                with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'dimensions_checks.txt'), 'a') as file:
                    file.write(f"{len(return_df[mask])} {len(return_df)} {INPUT_ROOTS_FOLDER}\n")
            File.write_df_on_file(return_df,
                                    filename=output_filename,
                                    fmt='pk')
        except Exception as e:
            with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                error_file.write(f"Error {e} in {INPUT_ROOTS_FOLDER}\n")

    @logger_decorator(logger)
    def do_work_parallel(self, binning, workers=1):
        input_runs_folder = DATA_LATACD_RAW_FOLDER_NAME
        output_runs_folder = DATA_LATACD_PROCESSED_FOLDER_NAME
        input_folder_list = os.listdir(input_runs_folder)
        if not os.path.exists(output_runs_folder):
            os.makedirs(output_runs_folder)
        else:
            if os.path.exists(os.path.join(output_runs_folder, 'pk')):
                output_folder_list = os.listdir(os.path.join(output_runs_folder, 'pk'))
            else:
                output_folder_list = os.listdir(output_runs_folder)
            prefix = ''.join(re.findall(r'\D+', input_folder_list[0]))
            for output_run in output_folder_list:
                if 'inputs_outputs' in output_run:
                    continue    
                numeric = re.search(r"\d+", output_run).group(0)
                run_folder = f'{prefix}{numeric}'
                if run_folder in input_folder_list or len(os.listdir(os.path.join(input_runs_folder, run_folder))) == 0:
                    input_folder_list.remove(run_folder)
        input_folder_list.sort(key=lambda x: int(re.search(r'\d+', x).group(0)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.create_df_norm, binning, os.path.join(output_runs_folder, re.search(r'\d+', run).group(0)), os.path.join(input_runs_folder, run)) for run in input_folder_list[:]]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing runs"):
                try:
                    future.result()
                except Exception as exc:
                    with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a') as error_file:
                        error_file.write(f"Error {exc}\n")

    def del_txt(self, folder):
        for file in glob.glob(f'{folder}/*/*.txt'):
            os.remove(file)


if __name__ == '__main__':
    fileSizes = os.path.join(DIR, 'ACD_tiles_size2.txt')
    arr = ACDReconRates()
    arr.del_txt(DATA_LATACD_RAW_FOLDER_NAME)
    arr.fill_dictSizes(fileSizes)
    arr.do_work_parallel(1, workers=3)
    print("... done, bye bye")
