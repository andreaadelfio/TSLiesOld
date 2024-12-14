'''
Main file to run the project pipeline
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 10/12/2024
TODO:
'''

# MARK: Imports
import itertools
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.plotter import Plotter
from modules.utils import Data, File, Time, Logger, logger_decorator
from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME, INPUTS_OUTPUTS_FILE_PATH
from modules.background import FFNNPredictor, PBNNPredictor, BNNPredictor, RNNPredictor, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor
from modules.spacecraft import SpacecraftOpener
from modules.catalog import CatalogReader
from modules.solar import SunMonitor
from modules.trigger import Trigger
from pipeline_config import y_cols, y_cols_raw, y_pred_cols, x_cols, x_cols_excluded, units, h_names

logger = Logger('Anomaly Detection Pipeline').get_logger()

# MARK: get_tiles_signal_df
@logger_decorator(logger)
def get_tiles_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(h_names=h_names, start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_catalog()
    runs_times = cr.get_runs_times()
    weeks_list = Time.get_week_from_datetime(runs_times)
    print(' done')
    return tile_signal_df, weeks_list

# MARK: get_sc_params_df
@logger_decorator(logger)
def get_sc_params_df(week, saa_exit_time):
    '''Get the spacecraft parameters dataframe'''
    sco = SpacecraftOpener()
    file = sco.get_sc_lat_weekly(week)
    sco.open(file, excluded_columns=['IN_SAA'])
    sc_params_df = sco.get_dataframe()
    sc_params_df = sco.get_good_quality_data(sc_params_df)
    sc_params_df, saa_exit_time = sco.add_saa_passage(sc_params_df, saa_exit_time)
    sc_params_df = sco.add_sun_occultation(sc_params_df)
    return sc_params_df, saa_exit_time

# MARK: get_solar_signal_df
@logger_decorator(logger)
def get_solar_signal_df(week):
    '''Get the solar signal dataframe from the GOES data'''
    tstart, tend = Time.get_datetime_from_week(week)
    sm = SunMonitor(tstart, tend)
    file_goes = sm.fetch_goes_data()
    solar_signal_df = sm.merge_goes_data(file_goes)
    return solar_signal_df

# MARK: get_inputs_outputs_df
@logger_decorator(logger)
def get_inputs_outputs_df():
    '''Get the inputs and outputs dataframe'''
    tile_signal_df, weeks_list = get_tiles_signal_df()
    saa_exit_time = 0
    inputs_outputs_list = []
    for week in tqdm([week for week in weeks_list if week not in ()], desc='Creating weekly datasets'):
        sc_params_df, saa_exit_time = get_sc_params_df(week, saa_exit_time)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        solar_signal_df = get_solar_signal_df(week)
        inputs_outputs = Data.merge_dfs(inputs_outputs, solar_signal_df)
        inputs_outputs['GOES_XRSA_HARD_EARTH_OCCULTED'] = (1 - inputs_outputs['SUN_IS_EARTH_OCCULTED']) * inputs_outputs['GOES_XRSA_HARD']
        inputs_outputs['GOES_XRSB_SOFT_EARTH_OCCULTED'] = (1 - inputs_outputs['SUN_IS_EARTH_OCCULTED']) * inputs_outputs['GOES_XRSB_SOFT']
        File.write_df_on_file(inputs_outputs, filename=f'{INPUTS_OUTPUTS_FILE_PATH}_w{week}', fmt='both')
        inputs_outputs_list.append(inputs_outputs)
    return pd.concat(inputs_outputs_list, ignore_index=True)

# MARK: run_model
@logger_decorator(logger)
def run_model(nn, inputs_outputs, y_cols, y_cols_raw, cols_pred, x_cols, hyperparams_combinations):
    '''Runs the specified model class with given hyperparameters'''
    for model_id, units_for_layers, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
        params = {'model_id': model_id, 'units_for_layers': units_for_layers, 'norm': norm, 'drop': drop, 'epochs': epochs,
                  'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
        nn.set_hyperparams(params)
        nn.create_model()
        history = nn.train()
        Plotter().plot_history(history)
        nn.update_summary()
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params)
        for start, end in [('2024-03-10 12:08:00', '2024-03-10 12:30:00'),
                           ('2024-03-28 20:50:00', '2024-03-28 21:10:00'),
                           ('2024-05-08 20:30:00', '2024-05-08 23:40:00'),
                           ('2024-05-11 01:00:00', '2024-05-11 03:00:00'),
                           ('2024-05-15 14:15:00', '2024-05-15 15:40:00'),
                           ('2024-05-08 01:00:00', '2024-05-08 05:00:00'), 
                           ('2024-06-20 22:35:00', '2024-06-20 23:40:00'), 
                           ('2024-06-23 05:35:00', '2024-06-23 14:40:00'), 
                           (str(inputs_outputs['datetime'].iloc[0]), str(inputs_outputs['datetime'].iloc[35000])),
                           (str(inputs_outputs['datetime'].iloc[35000]), str(inputs_outputs['datetime'].iloc[43000]))]:
            try:
                _, y_pred = nn.predict(start=start, end=end, write_bkg=False, save_plot=False)
                y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(cols_pred, y_cols)}).drop(columns=y_cols)
                tiles_df = Data.merge_dfs(Data.get_masked_dataframe(data=inputs_outputs, start=start, stop=end)[y_cols_raw + ['datetime', 'GOES_XRSA_HARD']], y_pred)
                Plotter(df=tiles_df, label='tiles').df_plot_tiles(y_cols, x_col='datetime', init_marker=',',
                                                                show=False, smoothing_key='pred', units=units)
                for col in y_cols_raw:
                    Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
                Plotter().plot_pred_true(tiles_df, cols_pred, y_cols_raw)
                Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params, (start, end))
            except Exception as e:
                print(f'Error {e} for {start}, {end}')

# MARK: run_trigger
def run_trigger(nn, inputs_outputs, y_cols, y_cols_raw, y_cols_pred, x_cols):
    '''Runs the trigger on the predicted data'''
    nn.set_model(model_path='data/background_prediction/0/model.keras')
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write_bkg=True, batch_size=1, save_predictions_plot=False)
    
    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs, y_pred)
    tiles_df = Data.get_masked_dataframe(data=tiles_df,
                                                  start='2024-06-20 22:35:00',
                                                  stop='2024-06-20 23:40:00', column='datetime').reset_index(drop=True)
    for col in y_cols_raw:
        Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
    Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(y_cols=y_cols, x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_cols_pred + y_cols_raw + ['GOES_XRSA_HARD_EARTH_OCCULTED']], init_marker = ',', show = True, smoothing_key='pred')
    thresholds = {'top': 6, 'Xpos': 10, 'Ypos': 7, 'Xneg': 7, 'Yneg': 7}
    merged_anomalies_list, triggs_df = Trigger().trigger(tiles_df, y_cols_raw, y_cols_pred, thresholds)
    support_vars = ['GOES_XRSA_HARD_EARTH_OCCULTED']
    tiles_df = Data.merge_dfs(tiles_df, triggs_df, on_column='datetime')
    Plotter(df = merged_anomalies_list).plot_anomalies(support_vars, thresholds, tiles_df, y_cols_raw, y_cols_pred, show=False, units=units)

########### Main ############
if __name__ == '__main__':
    x_cols = [col for col in x_cols if col not in x_cols_excluded]

    # MARK: Get inputs_outputs dataframe
    # inputs_outputs_df = get_inputs_outputs_df()
    inputs_outputs_df = File().read_dfs_from_weekly_pk_folder(start=0, stop=1000)
    inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
                                                  start='2024-06-20 22:35:00',
                                                  stop='2024-06-20 23:40:00')

    Plotter(df = inputs_outputs_df[['GOES_XRSA_HARD', 'GOES_XRSB_SOFT', 'MET', 'datetime']], label = 'Outputs').df_plot_tiles(
                                                    y_cols=y_cols,
                                                    x_col ='MET',
                                                    top_x_col='datetime',
                                                    excluded_cols = y_cols_raw + ['SUN_IS_EARTH_OCCULTED', 'MET', 'datetime', 'TIME_FROM_SAA', 'SAA_EXIT', 'GOES_XRSA_HARD_EARTH_OCCULTED', 'GOES_XRSB_SOFT_EARTH_OCCULTED'],
                                                    init_marker=',',
                                                    smoothing_key='smooth',
                                                    save = True,
                                                    units=units)
    Plotter.show()

    # MARK: Choose NN model
    nn = FFNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_pred_cols, False)
    # nn = PBNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_pred_cols, False)
    # nn = BNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_pred_cols, False)
    # nn = RNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_pred_cols, False)

    # MARK: Set NN hyperparameters
    hyperparams_combinations = {
        'units_for_layers' : ([90, 180], [90], [90], [90], [90], [90], [90], [70], [50], [30]),
        'epochs' : [5],
        'bs' : [1000],
        'do' : [0.02],
        'norm' : [0],
        'drop' : [0],
        'opt_name' : ['Adam'],
        'lr' : [None],
        'loss_type' : ['mae']
    }

    for params in nn.get_hyperparams_combinations(hyperparams_combinations, use_previous=False):
        nn.set_hyperparams(params)
        nn.create_model()
        history = nn.train()
        if history.history['loss'][-1] > 0.0040:
            continue
        Plotter().plot_history(history)
        nn.update_summary()
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params)
        for start, end in [('2024-03-10 12:08:00', '2024-03-10 12:30:00'),
                           ('2024-03-28 20:50:00', '2024-03-28 21:10:00'),
                           ('2024-05-08 20:30:00', '2024-05-08 23:40:00'),
                           ('2024-05-11 01:00:00', '2024-05-11 03:00:00'),
                           ('2024-05-15 14:15:00', '2024-05-15 15:40:00'),
                           ('2024-05-08 01:00:00', '2024-05-08 05:00:00'), 
                           ('2024-06-20 22:35:00', '2024-06-20 23:40:00'), 
                           ('2024-06-23 05:35:00', '2024-06-23 14:40:00'), 
                           (str(inputs_outputs_df['datetime'].iloc[0]), str(inputs_outputs_df['datetime'].iloc[35000])),
                           (str(inputs_outputs_df['datetime'].iloc[35000]), str(inputs_outputs_df['datetime'].iloc[43000]))]:
            nn.predict(start=start, end=end, mask_column='datetime', write_bkg=False, save_predictions_plot=True, support_variables=['SOLAR_a'])


    # MARK: Choose best model and trigger on the predicted data
    nn.set_model(model_path='data/background_prediction/0/model.keras')
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write_bkg=True, batch_size=1, save_predictions_plot=False)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs_df, y_pred)
    tiles_df = Data.get_masked_dataframe(data=tiles_df,
                                        start='2024-06-20 22:35:00',
                                        stop='2024-06-20 23:40:00', column='datetime').reset_index(drop=True)
    for col in y_cols_raw:
        Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
    Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(y_cols=y_cols, x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_pred + y_cols_raw + ['GOES_XRSA_HARD_EARTH_OCCULTED']], init_marker = ',', show = True, smoothing_key='pred')
    thresholds = {'top': 6, 'Xpos': 10, 'Ypos': 7, 'Xneg': 7, 'Yneg': 7}
    merged_anomalies_list, triggs_df = Trigger().trigger(tiles_df, y_cols_raw, y_pred, thresholds)
    support_vars = ['GOES_XRSA_HARD_EARTH_OCCULTED']
    tiles_df = Data.merge_dfs(tiles_df, triggs_df, on_column='datetime')
    Plotter(df = merged_anomalies_list).plot_anomalies(support_vars, thresholds, tiles_df, y_cols_raw, y_pred, show=False, units=units)
