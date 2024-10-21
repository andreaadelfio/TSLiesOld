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
        nn = FFNNPredictor(inputs_outputs, y_cols, x_cols, y_cols_raw, y_cols_pred)
        nn.set_scaler()
        nn.set_model(model_path='data/background_prediction/0/model.keras')
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write_bkg=True, batch_size=1, save_plot=False)
    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs, y_pred)
    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                               start='2024-03-21 23:45:22',
    #                                               stop='2024-05-21 23:59:28').reset_index(drop=True)
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_cols_pred + y_cols + ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], marker = ',', show = True, smoothing_key='pred')
    merged_anomalies_list = Trigger().trigger(tiles_df, y_cols_raw, y_cols_pred, threshold=3)
    # merged_anomalies_list = {"2701": {"top": {"changepoint": 2701, "stopping_time": 3164, "start_datetime": "2024-02-09 13:46:03", "stop_datetime": "2024-02-09 13:46:03", "significance": 0.008554283939547276, "max_significance": 0.010720111565244583, "sigma_val": 0.002848191337849441, "threshold": 3, "max_point": 2924}}, "38514": {"top": {"changepoint": 38514, "stopping_time": 39901, "start_datetime": "2024-02-10 03:59:00", "stop_datetime": "2024-02-10 03:59:00", "significance": 0.01022438562041527, "max_significance": 0.013547062336562913, "sigma_val": 0.002848191337849441, "threshold": 3, "max_point": 39392}}}
    support_vars = ['SOLAR_a_EARTH_OCCULTED']
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
    x_cols = [col for col in x_cols if col not in x_cols_excluded]
    inputs_outputs_df = File.read_dfs_from_weekly_pk_folder(start=819, stop=825)
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2024-02-08 05:30:22',
    #                                               stop='2024-09-11 09:15:28')
    
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], marker = ',', show = True, smoothing_key='smooth')
    run_trigger_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_trigger_with_median(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
