# legge lista di file root generati da lanciaReadRecon.py e
# produce un file root con:
# istogrammi dei rates per ogni tiles, e mediati sulle 5 facce
# istogrammi dei rates  normalizzati  per ogni tiles, e mediati sulle 5 facce

# usage:
# python makeReconRates.py #  --listFile LISTFILE (txt file with list of root files)  --binning BINNING (binning in seconds)  --outfile OUTFILE     (outrootfile, default: out.root) --acdSizesFile ACDSIZESFILE  (files with acd tiles area, default=ACD_tiles_size2.txt),  --t0 T0  (t0, default: 0.0)

# esempio:
# python ../ACDTransients/makeReconRates.py --listFile fileList --binning 1 --outfile out.root --t0 0 --acdSizesFile ../ACDTransients/ACD_tiles_size2.txt

# NB, la lista deve essere ordinata! (se  no la funzione get_time0_last non funziona... sarebbe da cambiare )
# es: ls outAcdReconRates_*_?_*.root >fileList
#     ls outAcdReconRates_*_??_*.root >>fileList


# from __future__ import print_function, division
import os
import argparse
import concurrent.futures
import logging
import ROOT
from tqdm import tqdm

logging.basicConfig(filename='mymakereconrate.log', level=logging.INFO)
logger = logging.getLogger(__name__)

dict_tileSize = {}


def get_time0_last(tree):

    # TODO: qua si assume che il tree sia ordinato (ossia la prima entry corrisponda al met minore e l'ultima a quella maggiore...)
    # bisognerebbe cercare max e min  indipendentemente dall'ordine...

    nEntries = tree.GetEntries()
    tree.GetEntry(0)
    time0 = tree.time

    tree.GetEntry(nEntries - 1)
    timeLast = tree.time

    return time0, timeLast


def fill_dictSizes(inFileName):
    f = open(inFileName, "r")

    for line in f:
        splitted_line = line.split()
        tile = int(splitted_line[0])
        area = float(splitted_line[1])
        dict_tileSize[tile] = area


def mediaSides(hist_dict, identityFunc):

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


def createTChain(rootfiles, treeName, path):
    chain = ROOT.TChain(treeName)
    separator = path.split('_')[-1]
    temp_dict = {}
    for line in rootfiles:
        if line.endswith('.root'):
            temp_dict[int(line.split(separator)[1].split('_')[0])] = line
    keys = list(temp_dict.keys())
    keys.sort()
    for key in keys:
        rootFile = f'{path}/{temp_dict[key]}'
        chain.Add(rootFile)
    return chain

def create_root(run, binning, output_filename, INPUT_ROOTS_FOLDER):
    list_file = os.listdir(INPUT_ROOTS_FOLDER)
    list_file.sort()
    output_rootfile = ROOT.TFile(output_filename, 'recreate')
    logger.info(f'Processing {output_filename}')
    myTree = createTChain(list_file, 'myTree', INPUT_ROOTS_FOLDER)

    time0, timeLast = get_time0_last(myTree)
    n_bins = int((timeLast - time0) / binning)
    identityFunc = ROOT.TF1("identityFunc", "1", time0, timeLast)

    # fill hist di tutti i triggers... serve per avere i rates normalizzati (i.e. acd occupancy)
    hist_triggers = ROOT.TH1F(
        "hist_triggers", "hist_triggers", n_bins, time0, timeLast)
    myString = 'time >> hist_triggers'
    myTree.Draw(myString, "", "goff")
    hist_triggers.Sumw2()
    # divido per larghezza bin => rate in Hz
    hist_triggers.Divide(identityFunc, binning)

    hist_dict = {}
    # histNorm_dict = {}

    for tileID in range(0, 89):
        hist_name = 'rate_tile'+str(tileID)
        hist_dict[tileID] = ROOT.TH1F(
            hist_name, hist_name, n_bins, time0, timeLast)

        string = 'time >> '+hist_name
        # cut="acdE_acdtile["+str(tileID)+"] >0.04"  # cut E>0.04!!!!!!!!!!!!
        cut = "acdE_acdtile["+str(tileID)+"] >0"

        myTree.Draw(string, cut, "goff")
        hist_dict[tileID].Sumw2()
        hist_dict[tileID].Divide(
            identityFunc, binning*dict_tileSize[tileID])
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

    hist_top, hist_Xpos, hist_Xneg, hist_Ypos, hist_Yneg = mediaSides(
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
    logger.info(f'Processing {output_filename} - done')

def do_work(binning):
    INPUT_RUNS_FOLDER = './data/LAT_ACD/complete/'
    OUTPUT_RUNS_FOLDER = './data/LAT_ACD/output runs/'
    input_folder_list = os.listdir(INPUT_RUNS_FOLDER)
    output_folder_list = os.listdir(OUTPUT_RUNS_FOLDER)
    for output_run in output_folder_list:
        if f'testOutputs_{output_run.split(".root")[0]}' in input_folder_list:
            input_folder_list.remove(f'testOutputs_{output_run.split(".root")[0]}')
    i = 0
    for run in input_folder_list:
        output_filename = f'{OUTPUT_RUNS_FOLDER}{run.split("_")[1]}.root'
        INPUT_ROOTS_FOLDER = INPUT_RUNS_FOLDER + run
        if len(os.listdir(INPUT_ROOTS_FOLDER)) > 0:
            print(f'{run} - {100 * (i+1)/len(input_folder_list)}%')
            create_root(run, binning, output_filename, INPUT_ROOTS_FOLDER)
            i += 1

def do_work_parallel(binning):
    INPUT_RUNS_FOLDER = './data/LAT_ACD/merged input runs/'
    OUTPUT_RUNS_FOLDER = './data/LAT_ACD/output runs/parallel/'
    input_folder_list = os.listdir(INPUT_RUNS_FOLDER)
    if not os.path.exists(OUTPUT_RUNS_FOLDER):
        os.makedirs(OUTPUT_RUNS_FOLDER)
    output_folder_list = os.listdir(OUTPUT_RUNS_FOLDER)
    for output_run in output_folder_list:
        run_folder = f'testOutputs_{output_run.split(".root")[0]}'
        root_dir = os.path.join(INPUT_RUNS_FOLDER, run_folder)
        if output_run != 'parallel':
            if run_folder in input_folder_list or len(os.listdir(root_dir)) == 0:
	            input_folder_list.remove(run_folder)

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_root, run, binning, f'{OUTPUT_RUNS_FOLDER}{run.split("_")[1]}.root', INPUT_RUNS_FOLDER + run) for run in input_folder_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing runs"):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')

def parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)

    parser.add_argument('--listFile', type=str,
                        help='txt file with list of root files ')
    parser.add_argument('--binning', type=float,  help='binning in seconds')
    parser.add_argument('--outfile', type=str,
                        help='outrootfile ', default='out.root')
    parser.add_argument('--acdSizesFile', type=str, default='ACD_tiles_size2.txt',
                        help='file with acd tiles area')
    parser.add_argument('--t0', type=float, default=0.,
                        help=' t0  ')

    return parser.parse_args()


if __name__ == '__main__':
    fileSizes = 'ACD_tiles_size2.txt'
    fill_dictSizes(fileSizes)
    # do_work(1)
    do_work_parallel(1)
    print("... done, bye bye")
