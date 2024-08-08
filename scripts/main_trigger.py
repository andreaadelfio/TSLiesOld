'''
Main file to run the project
Author: Andrea Adelfio
Created date: 24/06/2024
Modified date: 27/06/2024
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
from modules.trigger import trigger

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
    nn = FFNNPredictor(inputs_outputs, y_cols, x_cols)
    nn.set_scaler(inputs_outputs[x_cols])
    nn.set_model(model_path='data/background_prediction/0/model.keras')
    y_pred = File.read_df_from_file('data/background_prediction/0/pk/bkg')
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write=True)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs[y_cols_raw + x_cols + ['datetime', 'SUN_IS_OCCULTED']], y_pred)
    trigger(tiles_df, y_cols, y_cols_pred, threshold=3, bsize=180)

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
    tiles_df = Data.merge_dfs(inputs_outputs[y_cols_raw + x_cols + ['datetime', 'SUN_IS_OCCULTED']], y_pred)
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
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-21 23:45:22',
    #                                               stop='2023-12-21 23:59:28')
    
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], marker = ',', show = True, smoothing_key='smooth')
    x_cols = [col for col in x_cols if col not in x_cols_excluded]

    run_trigger_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_trigger_with_median(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)