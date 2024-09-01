'''
This module contains the class to manipulate the runs from Fermi.
'''
import os
import concurrent.futures
import re
import ROOT
from tqdm import tqdm
import pandas as pd
import numpy as np
try:
    from modules.plotter import Plotter
    from modules.config import DATA_LATACD_FOLDER_NAME, DATA_LATACD_INPUT_FOLDER_NAME
    from modules.utils import Logger, logger_decorator, File
except:
    from plotter import Plotter
    from config import DATA_LATACD_FOLDER_NAME, DATA_LATACD_INPUT_FOLDER_NAME
    from utils import Logger, logger_decorator, File


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
    def createTChain(self, rootfiles, treeName, path, end = None):
        chain = ROOT.TChain(treeName) # pylint: disable=maybe-no-member
        run = re.search(r"\d+", os.path.basename(path)).group(0)
        print(run)
        for line in rootfiles:
            if not line.endswith('.root'):
                rootfiles.remove(line)
        rootfiles.sort(key=lambda x: int(re.search(r"\d+", os.path.splitext(x)[0].split(run)[-1]).group(0)))
        for run in rootfiles[:end]:
            chain.Add(os.path.join(path, run))
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

    def create_root_df(self, binning, output_filename, INPUT_ROOTS_FOLDER):
        tiles_faces = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}
        list_file = os.listdir(INPUT_ROOTS_FOLDER)
        myTree = self.createTChain(list_file,'myTree', INPUT_ROOTS_FOLDER)
        dict_np = ROOT.RDataFrame(myTree).AsNumpy() # pylint: disable=maybe-no-member
        df = pd.DataFrame(dict_np)
        # df.to_csv('test.csv')

        notnull = df['acdE_acdtile'].apply(lambda x: any(x))
        df = df[notnull].reset_index(drop=True)
        if 'gltGemEngine' in df.columns:
            df = df[df['gltGemEngine'] != 3].reset_index(drop=True)

        time0, time_last = df['time'].min(), df['time'].max()
        n_bins = int((time_last - time0) / binning)
        bin_edges = np.linspace(time0, time_last, n_bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return_df = pd.DataFrame(bin_centers, columns=['MET'])

        # print('start')
        # import time
        # start = time.time()
        # acdE_acdtile = df['acdE_acdtile'].apply(lambda x: x[i] for i in range(89)).values.T
        acdE_acdtile = np.array([[i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15], i[16], i[17], i[18], i[19], i[20], i[21], i[22], i[23], i[24], i[25], i[26], i[27], i[28], i[29], i[30], i[31], i[32], i[33], i[34], i[35], i[36], i[37], i[38], i[39], i[40], i[41], i[42], i[43], i[44], i[45], i[46], i[47], i[48], i[49], i[50], i[51], i[52], i[53], i[54], i[55], i[56], i[57], i[58], i[59], i[60], i[61], i[62], i[63], i[64], i[65], i[66], i[67], i[68], i[69], i[70], i[71], i[72], i[73], i[74], i[75], i[76], i[77], i[78], i[79], i[80], i[81], i[82], i[83], i[84], i[85], i[86], i[87], i[88]] for i in df['acdE_acdtile'].values]).T
        # print(time.time() - start)
        # print(acdE_acdtile)
        
        for face, tiles in tiles_faces.items():
            face_rates = np.zeros(n_bins)
            face_rates_low = np.zeros(n_bins)
            face_rates_high = np.zeros(n_bins)
            
            for tile in range(tiles[0], tiles[-1]):
                times_with_energy = df['time'][acdE_acdtile[tile] > 0].values
                times_with_energy_low = df['time'][(acdE_acdtile[tile] > 0) & (acdE_acdtile[tile] < 5)].values
                times_with_energy_high = df['time'][acdE_acdtile[tile] > 5].values
                
                counts, _ = np.histogram(times_with_energy, bins=bin_edges)
                counts_low, _ = np.histogram(times_with_energy_low, bins=bin_edges)
                counts_high, _ = np.histogram(times_with_energy_high, bins=bin_edges)
                norm = 1 / binning / self.dict_tileSize[tile]
                face_rates += counts * norm
                face_rates_low += counts_low * norm
                face_rates_high += counts_high * norm

            return_df[face] = face_rates / (tiles[-1] - tiles[0])
            return_df[f"{face}_low"] = face_rates_low / (tiles[-1] - tiles[0])
            return_df[f"{face}_high"] = face_rates_high / (tiles[-1] - tiles[0])
        
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
                                fmt='both')

    @logger_decorator(logger)
    def do_work_parallel(self, binning, workers=1):
        input_runs_folder = DATA_LATACD_INPUT_FOLDER_NAME
        output_runs_folder = DATA_LATACD_FOLDER_NAME
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
                numeric = re.search(r"\d+", output_run).group(0)
                run_folder = f'{prefix}{numeric}'
                if run_folder in input_folder_list or len(os.listdir(os.path.join(input_runs_folder, run_folder))) == 0:
                    input_folder_list.remove(run_folder)

        input_folder_list.sort(key=lambda x: int(re.search(r'\d+', x).group(0)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.create_root_df, binning, os.path.join(output_runs_folder, re.search(r'\d+', run).group(0)), os.path.join(input_runs_folder, run)) for run in input_folder_list[:]]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing runs"):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')


if __name__ == '__main__':
    fileSizes = 'ACD_tiles_size2.txt'
    arr = ACDReconRates()
    arr.fill_dictSizes(fileSizes)
    arr.do_work_parallel(1, workers=1)
    print("... done, bye bye")
