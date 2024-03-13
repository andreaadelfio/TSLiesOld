"""Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 11/03/2024
TODO:
"""
import pandas as pd
from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter
from sunmonitor import SunMonitor
from utils import Data, File
from config import SOLAR_FILE_PATH, TILE_SIGNAL_FILE_PATH, SC_FILE_PATH, INPUTS_OUTPUTS_FILE_PATH
from nn import NN

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_df_from_file(INPUTS_OUTPUTS_FILE_PATH)
    if inputs_outputs_df is None:
        ########## BAT Catalog ##########
        print('BAT Catalog...', end='')
        tile_signal_df = File.read_df_from_file(TILE_SIGNAL_FILE_PATH)
        if tile_signal_df is None:
            cr = CatalogReader(start = 0, end = 100)
            tile_signal_df = cr.get_signal_df_from_catalog()
            tile_signal_df = cr.add_smoothing(tile_signal_df)
            File.write_df_on_file(tile_signal_df, TILE_SIGNAL_FILE_PATH)
            runs_times = cr.get_runs_times() # ?????????
            # File.write_on_file('data/runs_times.txt', runs_times)
        else:
            runs_times = tile_signal_df['datetime']
        
        print(' done')
        ############## GOES ##############
        print('GOES data...', end='')
        solar_signal_df = File.read_df_from_file(SOLAR_FILE_PATH)
        if solar_signal_df is None:
            tstart, tend = list(runs_times.values())[0][0], list(runs_times.values())[-1][1]
            sm = SunMonitor(tstart, tend)
            file_goes = sm.fetch_goes_data()
            solar_signal_df = sm.find_goes_data(file_goes)
            solar_signal_df = Data.filter_dataframe_with_run_times(solar_signal_df, runs_times)
            File.write_df_on_file(solar_signal_df, SOLAR_FILE_PATH)
        
        print(' done')
        ############## LAT ##############
        print('LAT data...', end='')

        sc_params_df = File.read_df_from_file(SC_FILE_PATH)
        if sc_params_df is None:
            sco = SpacecraftOpener()
            sco.open(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
            sc_params_df = sco.get_dataframe()
            sc_params_df = Data.filter_dataframe_with_run_times(sc_params_df, runs_times)
            File.write_df_on_file(sc_params_df, SC_FILE_PATH)

        print(' done')
        ############# MERGE #############
        print('MERGE...', end='')
        inputs_df = Data.merge_dfs(sc_params_df, solar_signal_df)
        inputs_outputs_df = Data.merge_dfs(tile_signal_df, inputs_df)
        File.write_df_on_file(inputs_outputs_df)

        print(' done')
        Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
        Plotter(df = inputs_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)
        Plotter.show()

    ########### Plotting ############
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-05 09:35:00', stop='2023-12-05 11:35:00')
    # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')
    # print('Plotting...', end='')
    Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
    Plotter.show()
    # print(' done')

    ############## NN ###############
    # nn = NN(inputs_outputs_df)
    # nn.train(loss_type='mae', units=2048, epochs=10, lr=0.0008, bs=2048, do=0.02)
    # nn.predict()
    
    # df_ori = pd.read_pickle("./" + 'frg_' + '.pk')
    # y_pred = pd.read_pickle("./" + 'bkg_' + '.pk')

    # # Plotter(df = df_ori, label = 'df_ori').df_plot_tiles(x_col = 'timestamp', excluded_cols = ['START', 'met'], marker = ',', show = False, with_smooth = True)
    # # Plotter(df = y_pred, label = 'y_pred').df_plot_tiles(x_col = 'timestamp', excluded_cols = ['START', 'met'], marker = ',', show = False, with_smooth = True)
    # nn.plot(det_rng='top')
    # nn.plot(det_rng='Xpos')
    # nn.plot(det_rng='Xneg')
    # nn.plot(det_rng='Ypos')
    # nn.plot(det_rng='Yneg')
    # Plotter.show()
