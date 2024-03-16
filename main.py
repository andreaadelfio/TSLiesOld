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
from config import SOLAR_FILE_PATH, TILE_SIGNAL_FILE_PATH, SC_FILE_PATH, INPUTS_OUTPUTS_FILE_PATH, MODEL_NN_FOLDER_NAME
from nn import NN

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_df_from_file(INPUTS_OUTPUTS_FILE_PATH)
    if inputs_outputs_df is None:
        ########## BAT Catalog ##########
        print('BAT Catalog...', end='')
        tile_signal_df = File.read_df_from_file(TILE_SIGNAL_FILE_PATH)
        if tile_signal_df is None:
            cr = CatalogReader(start = 0, end = 400)
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
            # sco.open(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
            sco.open(excluded_columns = ['STOP'])
            sc_params_df = sco.get_dataframe()
            sc_params_df = Data.filter_dataframe_with_run_times(sc_params_df, runs_times)
            File.write_df_on_file(sc_params_df, SC_FILE_PATH)

        print(' done')
        ############# MERGE #############
        print('MERGE...', end='')
        tile_signal_df = Data.merge_dfs(tile_signal_df, solar_signal_df)
        inputs_outputs_df = Data.merge_dfs(tile_signal_df, sc_params_df)
        File.write_df_on_file(inputs_outputs_df)

        print(' done')
        Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
        Plotter(df = sc_params_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)
        Plotter.show()

    ########### Plotting ############
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-05 18:10:00', stop='2023-12-10 18:40:00')
    # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')
    # print('Plotting...', end='')
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
    # Plotter.show()
    # print(' done')

    ############## NN ###############
    nn = NN(inputs_outputs_df)
    for units in [40]:
        for bs in [1000]:
            for do in [0.02]:
                for opt_name in ['Adam']:
                    # nn.train(loss_type='mae', units=units, epochs=500, lr=0.0008, bs=bs, do=do, opt_name=opt_name)
                    # nn.predict()
                    df_ori = pd.read_pickle(MODEL_NN_FOLDER_NAME + '/frg' + '.pk')
                    y_pred = pd.read_pickle(MODEL_NN_FOLDER_NAME + '/bkg' + '.pk')
                    y_pred['datetime'] = y_pred['timestamp']
                    cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
                    y_pred = y_pred.assign(**{f"{col}_smooth": y_pred[col] for col in cols})
                    inputs_outputs_df = Data.merge_dfs(inputs_outputs_df[[col for col in inputs_outputs_df.columns if '_smooth' not in col]], y_pred[['top_smooth','Xpos_smooth','Xneg_smooth','Ypos_smooth','Yneg_smooth', 'datetime']])
                    Plotter(df = inputs_outputs_df, label = 'inputs_outputs_df').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg', 'SOLAR', 'top_smooth','Xpos_smooth','Xneg_smooth','Ypos_smooth','Yneg_smooth']], marker = ',', show = False, with_smooth = True)
                    # nn.plot(df_ori, y_pred, det_rng='top')
                    nn.plot(df_ori, y_pred, det_rng='Xpos')
                    # nn.plot(df_ori, y_pred, det_rng='Xneg')
                    # nn.plot(df_ori, y_pred, det_rng='Ypos')
                    # nn.plot(df_ori, y_pred, det_rng='Yneg')
                    Plotter.show()
