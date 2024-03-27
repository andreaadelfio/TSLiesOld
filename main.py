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
from scripts.nn import NN
# from nn import MedianKNeighborsRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.neighbors import KNeighborsRegressor

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
                     'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'START', 'IN_SAA', 'STOP', 'LIVETIME'])
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
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-05 18:10:00', stop='2023-12-10 18:40:00')
    # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')
    # print('Plotting...', end='')
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
    # Plotter.show()
    # print(' done')

    ############## NN ###############
    col_range = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO', 'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ',
                    'DEC_SCZ', 'RA_SCX', 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN', 'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2', 'SOLAR']
    nn = NN(inputs_outputs_df, col_range, col_selected)
    units_1_values = [0, 50, 70, 90, 120]
    units_2_values = [0, 30, 50, 70, 90, 120]
    units_3_values = [0, 10, 30, 50, 70, 90, 120]
    epochs_values = [150]
    bs_values = [1000]
    do_values = [0.02]
    norm_1_values = [0, 1]
    drop_1_values = [0, 1]
    norm_2_values = [0, 1]
    drop_2_values = [0, 1]
    norm_3_values = [0, 1]
    drop_3_values = [0, 1]
    opt_name_values = ['Adam']
    lr_values = [0.00009]
    loss_type_values = ['mae', 'mse']
    hyperparams_combinations = list(itertools.product(units_1_values, units_2_values, units_3_values, epochs_values, bs_values, do_values, norm_1_values, drop_1_values, norm_2_values, drop_2_values, norm_3_values, drop_3_values, opt_name_values, lr_values, loss_type_values))
    hyperparams_combinations = nn.trim_hyperparams_combinations(hyperparams_combinations)

    for model_id, (units_1, units_2, units_3, epochs, bs, do, norm_1, drop_1, norm_2, drop_2, norm_3, drop_3, opt_name, lr, loss_type) in enumerate(hyperparams_combinations):
        params = {'model_id': model_id, 'units_1': units_1, 'units_2': units_2, 'units_3': units_3, 'epochs': epochs, 'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr,
                  'loss_type': loss_type, 'norm_1': norm_1, 'drop_1': drop_1, 'norm_2': norm_2, 'drop_2': drop_2, 'norm_3': norm_3, 'drop_3': drop_3}
        params_string = '_'.join([f'{k}_{v}' for k, v in params.items()])
        nn.set_hyperparams(params)
        nn.create_model()
        nn.train()
        nn.update_summary()
        nn.predict(start=50000, end=70000)
        df_ori = pd.read_pickle(MODEL_NN_FOLDER_NAME +
                                f'/{params_string}/frg_{params_string}.pk')
        y_pred = pd.read_pickle(MODEL_NN_FOLDER_NAME +
                                f'/{params_string}/bkg_{params_string}.pk')
        y_pred['datetime'] = y_pred['timestamp']
        cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
        y_pred = y_pred.assign(
            **{f"{col}_smooth": y_pred[col] for col in cols})
        inputs_outputs_df = Data.merge_dfs(inputs_outputs_df[[col for col in inputs_outputs_df.columns if '_smooth' not in col]], y_pred[[
                                           'top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth', 'datetime']])
        Plotter(df=inputs_outputs_df, label='inputs_outputs_df').df_plot_tiles(x_col='datetime', excluded_cols=[col for col in inputs_outputs_df.columns if col not in [
            'top', 'Xpos', 'Xneg', 'Ypos', 'Yneg', 'SOLAR', 'top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']], marker=',', show=False, with_smooth=True)
        # nn.plot(df_ori, y_pred, det_rng='top')
        nn.plot(df_ori, y_pred, det_rng='Xpos')
        # nn.plot(df_ori, y_pred, det_rng='Xneg')
        # nn.plot(df_ori, y_pred, det_rng='Ypos')
        # nn.plot(df_ori, y_pred, det_rng='Yneg')
        Plotter.save(MODEL_NN_FOLDER_NAME, params)

    # col_range = ['top','Xpos','Xneg','Ypos','Yneg']
    # col_selected = ['SC_POSITION_0','SC_POSITION_1','SC_POSITION_2','LAT_GEO','LON_GEO','RAD_GEO','RA_ZENITH','DEC_ZENITH','B_MCILWAIN','L_MCILWAIN','GEOMAG_LAT','LAMBDA','IN_SAA','RA_SCZ','DEC_SCZ','RA_SCX','DEC_SCX','RA_NPOLE','DEC_NPOLE','ROCK_ANGLE','LAT_MODE','LAT_CONFIG','DATA_QUAL','LIVETIME','QSJ_1','QSJ_2','QSJ_3','QSJ_4','RA_SUN','DEC_SUN','SC_VELOCITY_0','SC_VELOCITY_1','SC_VELOCITY_2','SOLAR']
    # y = inputs_outputs_df[col_range].astype('float32')
    # X = inputs_outputs_df[col_selected].astype('float32')
    # n_points = -1
    # clf = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=20)).fit(X, y)
    # pred_df = clf.predict(X[:n_points])
    # plt.figure()
    # plt.plot(inputs_outputs_df['datetime'][:n_points], y.iloc[:n_points, 0])
    # plt.plot(inputs_outputs_df['datetime'][:n_points], pred_df[:n_points, 0])
    # plt.show()
    # plt.plot(inputs_outputs_df['datetime'][:2000], pred_df[0, :2000])
