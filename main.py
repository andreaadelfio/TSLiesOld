"""Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 20/03/2024
TODO:
"""
import itertools
import pandas as pd
from scripts.spacecraftopener import SpacecraftOpener
from scripts.catalogreader import CatalogReader
from scripts.plotter import Plotter
from scripts.sunmonitor import SunMonitor
from scripts.utils import Data, File
from scripts.config import SOLAR_FILE_PATH, TILE_SIGNAL_FILE_PATH, SC_FILE_PATH, INPUTS_OUTPUTS_FILE_PATH, MODEL_NN_FOLDER_NAME
from scripts.nn import NN, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_df_from_file(INPUTS_OUTPUTS_FILE_PATH)
    if inputs_outputs_df is None:
        ########## BAT Catalog ##########
        print('BAT Catalog...', end='')
        tile_signal_df = File.read_df_from_file(TILE_SIGNAL_FILE_PATH)
        if tile_signal_df is None:
            cr = CatalogReader(start=0, end=-1)
            tile_signal_df = cr.get_signal_df_from_catalog()
            tile_signal_df = cr.add_smoothing(tile_signal_df)
            File.write_df_on_file(tile_signal_df, TILE_SIGNAL_FILE_PATH)
            runs_times = cr.get_runs_times()  # ?????????
            File.write_on_file('data/runs_times.txt', runs_times)
        else:
            runs_times = {'event': (tile_signal_df['datetime'][0], tile_signal_df['datetime'][len(
                tile_signal_df['datetime'])-1])}

        print(' done')
        ############## GOES ##############
        print('GOES data...', end='')
        solar_signal_df = File.read_df_from_file(SOLAR_FILE_PATH)
        if solar_signal_df is None:
            tstart, tend = list(runs_times.values())[
                0][0], list(runs_times.values())[-1][1]
            sm = SunMonitor(tstart, tend)
            file_goes = sm.fetch_goes_data()
            solar_signal_df = sm.find_goes_data(file_goes)
            solar_signal_df = Data.filter_dataframe_with_run_times(
                solar_signal_df, runs_times)
            File.write_df_on_file(solar_signal_df, SOLAR_FILE_PATH)

        print(' done')
        ############## LAT ##############
        print('LAT data...', end='')

        sc_params_df = File.read_df_from_file(SC_FILE_PATH)
        if sc_params_df is None:
            sco = SpacecraftOpener()
            # sco.open(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
            sco.open(excluded_columns=[
                     'LAT_MODE', 'LAT_CONFIG', 'START', 'IN_SAA', 'STOP', 'LIVETIME'])
            sc_params_df = sco.get_dataframe()
            sc_params_df = Data.filter_dataframe_with_run_times(
                sc_params_df, runs_times)
            File.write_df_on_file(sc_params_df, SC_FILE_PATH)

        print(' done')
        ############# MERGE #############
        print('MERGE...', end='')
        tile_signal_df = Data.merge_dfs(tile_signal_df, solar_signal_df)
        inputs_outputs_df = Data.merge_dfs(tile_signal_df, sc_params_df)
        File.write_df_on_file(inputs_outputs_df)

        print(' done')
        # Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
        # Plotter(df = sc_params_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)
        # Plotter.show()

    ########### Plotting ############
    ##### !! implementare data quality != 0
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-07 04:00:00', stop='2023-12-08 04:00:00')
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-14 04:00:00', stop='2023-12-15 04:00:00')
    # inputs_outputs_df = Data.get_good_quality(inputs_outputs_df)
    # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')
    # print('Plotting...', end='')
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth', 'START', 'MET', 'top', 'Xpos', 'Xneg', 'Ypos', 'Yneg', ], marker = ',', show = True, smoothing_key='smooth')
    # Plotter.show()
    # print(' done')

    # ############## NN ###############
    col_range_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    # # col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    col_range = ['Xpos']
    col_pred = [col + '_pred' for col in col_range_raw]
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO', 'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ',
                    'DEC_SCZ', 'RA_SCX', 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN', 'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2', 'SOLAR']


    # col_selected = [['RA_SUN', 'DEC_SUN', 'SOLAR']]
    
    # for col in ['datetime']:
    # Plotter(df = inputs_outputs_df[['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg', 'top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth'] + ['datetime']], label = 'counts_tiles_solar_activity').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = False, smoothing_key='smooth')
    # for col in col_selected:
    #     Plotter(df = inputs_outputs_df[['Xpos', 'Xpos_smooth', 'datetime'] + col], label = col[0] + '_datetime_solar_activity').df_plot_tiles_for_pres(x_col = 'datetime', excluded_cols = ['top_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth', 'START', 'MET', 'top', 'Xneg', 'Ypos', 'Yneg'], marker = ',', show = False, smoothing_key='smooth')
    #     Plotter(df = inputs_outputs_df[['Xpos', 'Xpos_smooth', 'datetime'] + col], label = col[0] + '_Xpos_solar_activity').df_plot_corr_tiles(x_col = 'Xpos', excluded_cols = [], marker = '.', ms = 0.5, lw = 0, show = False, smoothing_key='smooth')
    # Plotter().show()

    # nn = NN(inputs_outputs_df, col_range, col_selected)
    # units_1_values = [90]
    # units_2_values = [90]
    # units_3_values = [30, 70]
    # epochs_values = [60]
    # bs_values = [1000]
    # do_values = [0.02]
    # norm_values = [0]
    # drop_values = [0]
    # opt_name_values = ['Adam']
    # lr_values = [0.00009]
    # loss_type_values = ['mae']
    # hyperparams_combinations = list(itertools.product(units_1_values, units_2_values, units_3_values, norm_values, drop_values, epochs_values, bs_values, do_values, opt_name_values, lr_values, loss_type_values))
    # if True:
    #     hyperparams_combinations = nn.trim_hyperparams_combinations(hyperparams_combinations)
    # else:
    #     hyperparams_combinations = nn.use_previous_hyperparams_combinations(hyperparams_combinations)
        
    # for model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
    #     params = {'model_id': model_id, 'units_1': units_1, 'units_2': units_2, 'units_3': units_3, 'norm': norm, 'drop': drop, 'epochs': epochs, 'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
    #     nn.set_hyperparams(params)
    #     nn.create_model()
    #     history = nn.train()
    #     Plotter().plot_history(history, 'loss')
    #     Plotter().plot_history(history, 'accuracy')
    #     nn.update_summary()
    #     Plotter.save(MODEL_NN_FOLDER_NAME, params)
    #     for start, end in [(60000, 73000)]:#, (73000, -1)]:
    #         df_ori, y_pred = nn.predict(start=start, end=end)

    #         y_pred = y_pred.assign(**{col: y_pred[col_init] for col, col_init in zip(col_pred, col_range)}).drop(columns=col_range)
    #         tiles_df = Data.merge_dfs(inputs_outputs_df[start:end][col_range_raw + ['datetime', 'SOLAR']], y_pred)
    #         Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',', show=False, smoothing_key='pred')
    #         for col in col_range_raw:
    #             Plotter().plot_tile(tiles_df, det_rng=col, smoothing_key = 'pred')
    #         print(tiles_df.columns.tolist())
    #         Plotter().plot_pred_true(tiles_df, col_pred, col_range_raw)
    #         Plotter.save(MODEL_NN_FOLDER_NAME, params, (start, end))

    # Plotter().show_models_params(MODEL_NN_FOLDER_NAME, features_dict={'1 layer': {'units_1': 10, 'units_2': 10}, '2 layers': {'units_1': 10}, '3 layers': {}})
    # Plotter().show_models_params(MODEL_NN_FOLDER_NAME, features_dict={'1 layer': {'units_1': 10, 'units_2': 10}, '2 layers': {'units_1': 10}, '3 layers': {}})
    # Plotter.save(MODEL_NN_FOLDER_NAME)

    multi_reg = MultiMedianKNeighborsRegressor(inputs_outputs_df, col_range, col_selected)
    # multi_reg = MultiMeanKNeighborsRegressor(inputs_outputs_df, col_range, col_selected)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 63000
    df_ori, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=col_range)
    Plotter().plot_tile_knn(inputs_outputs_df[start:end].reset_index(), y_pred, det_rng='Xpos')
    Plotter().show()
    # Plotter.save(MODEL_NN_FOLDER_NAME)
