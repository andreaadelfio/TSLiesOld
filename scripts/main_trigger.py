'''
Main file to run the project
Author: Andrea Adelfio
Created date: 24/06/2024
Modified date: 13/09/2024
TODO:
'''

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.background import RNNPredictor, FFNNPredictor
from modules.utils import Data, File
from modules.plotter import Plotter
from modules.trigger import Trigger

from scripts.main_config import y_cols, y_cols_raw, y_pred_cols, x_cols, x_cols_excluded


def run_trigger_rnn(inputs_outputs, y_cols, y_cols_raw, y_cols_pred, x_cols):
    '''Runs the model'''
    nn = RNNPredictor(inputs_outputs, y_cols, x_cols, timestep=10)
    nn.set_scaler(inputs_outputs[x_cols])
    nn.set_model(model_path='data/background_prediction/0/model.keras')
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write=True, batched=True)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs[y_cols_raw + x_cols + ['datetime', 'SUN_IS_OCCULTED']], y_pred)
    trigger(tiles_df, y_cols, y_cols_pred, threshold=5, bsize=50)

def run_trigger_ffnn(inputs_outputs, y_cols, y_cols_raw, y_cols_pred, x_cols):
    '''Runs the model'''
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        nn = FFNNPredictor(inputs_outputs, y_cols, x_cols)
        nn.set_scaler(inputs_outputs[x_cols])
        nn.set_model(model_path='data/background_prediction/0/model.keras')
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write=True)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs[y_cols_raw + x_cols + ['MET', 'datetime', 'SUN_IS_OCCULTED']], y_pred)
    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                               start='2024-03-21 23:45:22',
    #                                               stop='2024-05-21 23:59:28').reset_index(drop=True)
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_cols_pred + y_cols + ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], marker = ',', show = True, smoothing_key='pred')
    merged_anomalies_list = Trigger().trigger(tiles_df, y_cols_raw, y_cols_pred, threshold=3)
    support_vars = ['SUN_IS_OCCULTED']
    # merged_anomalies_list = {"37129": {"top": {"changepoint": 37129, "stopping_time": 38484, "start_datetime": "2024-05-11 06:32:44", "stop_datetime": "2024-05-11 06:55:19", "significance": 0.0085756260751932, "sigma_val": 0.0028565211994213773, "threshold": 3}, "Yneg": {"changepoint": 37377, "stopping_time": 38364, "start_datetime": "2024-05-11 06:36:52", "stop_datetime": "2024-05-11 06:53:19", "significance": 0.012213942266409521, "sigma_val": 0.004061499439506119, "threshold": 3}}, "38918": {"top": {"changepoint": 38918, "stopping_time": 40156, "start_datetime": "2024-05-11 10:46:34", "stop_datetime": "2024-05-11 11:07:15", "significance": 0.008576215608834366, "sigma_val": 0.0028565211994213773, "threshold": 3}, "Ypos": {"changepoint": 38918, "stopping_time": 39893, "start_datetime": "2024-05-11 10:46:34", "stop_datetime": "2024-05-11 11:02:52", "significance": 0.011848404045314686, "sigma_val": 0.0039425431712174636, "threshold": 3}, "Yneg": {"changepoint": 38918, "stopping_time": 40257, "start_datetime": "2024-05-11 10:46:34", "stop_datetime": "2024-05-11 11:08:58", "significance": 0.012214286099164164, "sigma_val": 0.004061499439506119, "threshold": 3}}, "51300": {"Xpos": {"changepoint": 51300, "stopping_time": 52464, "start_datetime": "2024-05-12 16:22:04", "stop_datetime": "2024-05-12 16:41:52", "significance": 0.015239216054225738, "sigma_val": 0.005071540549375602, "threshold": 3}}, "31910": {"Xpos": {"changepoint": 31910, "stopping_time": 34072, "start_datetime": "2024-05-10 20:04:19", "stop_datetime": "2024-05-10 20:40:32", "significance": 0.03356282299604148, "sigma_val": 0.005071540549375602, "threshold": 3}}, "39404": {"Xpos": {"changepoint": 39404, "stopping_time": 40158, "start_datetime": "2024-05-11 10:54:43", "stop_datetime": "2024-05-11 11:07:17", "significance": 0.01522203774085595, "sigma_val": 0.005071540549375602, "threshold": 3}}, "42884": {"Xpos": {"changepoint": 42884, "stopping_time": 43042, "start_datetime": "2024-05-11 11:53:37", "stop_datetime": "2024-05-11 11:56:15", "significance": 0.023534797052033103, "sigma_val": 0.005071540549375602, "threshold": 3}}, "53596": {"Xpos": {"changepoint": 53596, "stopping_time": 54111, "start_datetime": "2024-05-12 22:02:36", "stop_datetime": "2024-05-12 22:11:12", "significance": 0.015273355356105831, "sigma_val": 0.005071540549375602, "threshold": 3}}, "75542": {"Xpos": {"changepoint": 75542, "stopping_time": 78346, "start_datetime": "2024-05-13 08:49:46", "stop_datetime": "2024-05-13 09:40:37", "significance": 0.026763632736760495, "sigma_val": 0.005071540549375602, "threshold": 3}}, "98000": {"Xpos": {"changepoint": 98000, "stopping_time": 100515, "start_datetime": "2024-05-14 12:44:41", "stop_datetime": "2024-05-14 13:50:45", "significance": 0.015222841134789108, "sigma_val": 0.005071540549375602, "threshold": 3}}, "112283": {"Xpos": {"changepoint": 112283, "stopping_time": 112560, "start_datetime": "2024-05-14 23:05:30", "stop_datetime": "2024-05-14 23:10:49", "significance": 0.015276818029942851, "sigma_val": 0.005071540549375602, "threshold": 3}}, "16024": {"Xneg": {"changepoint": 16024, "stopping_time": 18817, "start_datetime": "2024-05-09 09:43:19", "stop_datetime": "2024-05-09 10:29:52", "significance": 0.01246526055052961, "sigma_val": 0.004152519661779955, "threshold": 3}}, "33218": {"top": {"changepoint": 33297, "stopping_time": 34072, "start_datetime": "2024-05-10 20:27:26", "stop_datetime": "2024-05-10 20:40:32", "significance": 0.017273283430960597, "sigma_val": 0.0028565211994213773, "threshold": 3}, "Ypos": {"changepoint": 33218, "stopping_time": 34072, "start_datetime": "2024-05-10 20:26:07", "stop_datetime": "2024-05-10 20:40:32", "significance": 0.026011429625652463, "sigma_val": 0.0039425431712174636, "threshold": 3}, "Yneg": {"changepoint": 33290, "stopping_time": 34072, "start_datetime": "2024-05-10 20:27:19", "stop_datetime": "2024-05-10 20:40:32", "significance": 0.029424657154626756, "sigma_val": 0.004061499439506119, "threshold": 3}}}
    Plotter(df = merged_anomalies_list).plot_anomalies(support_vars, tiles_df, y_cols_raw, y_cols_pred)

def run_trigger_with_median(inputs_outputs, y_cols, y_cols_raw, y_cols_pred, x_cols):
    '''Runs the model'''
    nn = FFNNPredictor(inputs_outputs, y_cols, x_cols)
    nn.set_scaler(inputs_outputs[x_cols])
    nn.set_model(model_path='data/background_prediction/0/model.keras')
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write=True)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs[y_cols_raw + x_cols + ['MET', 'datetime', 'SUN_IS_OCCULTED']], y_pred)
    for col in y_cols_raw:
        tiles_df[f'{col}_dif'] = tiles_df[col] - tiles_df[f'{col}_pred']
    
    median = np.median(tiles_df[[f'{col}_dif' for col in y_cols_raw]], axis=1)

    mean_median = median.mean()
    std_median = median.std()
    normalized_median = (median - mean_median) / std_median
    for col in y_cols:
        std_diff = tiles_df[f'{col}_dif'].std()
        tiles_df[f'{col}_pred'] = tiles_df[f'{col}_pred'] + normalized_median * std_diff 

    Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')
    trigger(tiles_df, y_cols, y_cols_pred, threshold=5, bsize=100)


if __name__ == '__main__':
    inputs_outputs_df = File.read_dfs_from_pk_folder()
    # inputs_outputs_df['MEDIAN'] = np.median(inputs_outputs_df[['top', 'Xneg', 'Ypos', 'Yneg']], axis=1)
    inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
                                                  start='2024-02-08 05:30:22',
                                                  stop='2024-10-06 09:15:28')
    
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], marker = ',', show = True, smoothing_key='smooth')
    x_cols = [col for col in x_cols if col not in x_cols_excluded]

    run_trigger_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_trigger_with_median(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)