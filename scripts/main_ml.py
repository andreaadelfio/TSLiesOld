"""Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 27/06/2024
TODO:
"""

import itertools
import sys
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.plotter import Plotter
from modules.utils import Data, File
from modules.config import MODEL_NN_FOLDER_NAME
from modules.nn import NN, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor, get_feature_importance

from config import y_cols, y_cols_raw, y_pred_cols, x_cols, x_cols_excluded

def run_nn(inputs_outputs, cols_range, cols_range_raw, cols_pred, cols_selected):
    '''Runs the neural network model'''
    # inputs_outputs['MEDIAN'] = np.median(inputs_outputs[['top', 'Xneg', 'Ypos', 'Yneg']], axis=1)

    nn = NN(inputs_outputs, cols_range, cols_selected)
    units_1_values = [90]
    units_2_values = [90]
    units_3_values = [70]
    epochs_values = [100]
    bs_values = [1000]
    do_values = [0.02]
    norm_values = [0]
    drop_values = [0]
    opt_name_values = ['Adam']
    lr_values = [0.00009]
    loss_type_values = ['mae']

    Plotter().plot_correlation_matrix(inputs_outputs, show=False, save=True)
    
    hyperparams_combinations = list(itertools.product(units_1_values, units_2_values,
                                                      units_3_values, norm_values,
                                                      drop_values, epochs_values,
                                                      bs_values, do_values, opt_name_values,
                                                      lr_values, loss_type_values))
    hyperparams_combinations = nn.trim_hyperparams_combinations(hyperparams_combinations)
    # hyperparams_combinations = nn.use_previous_hyperparams_combinations(hyperparams_combinations)
    for model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
        params = {'model_id': model_id, 'units_1': units_1, 'units_2': units_2,
                  'units_3': units_3, 'norm': norm, 'drop': drop, 'epochs': epochs,
                  'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
        nn.set_hyperparams(params)
        nn.create_model()
        history = nn.train()
        Plotter().plot_history(history, 'loss')
        Plotter().plot_history(history, 'accuracy')
        nn.update_summary()
        Plotter.save(MODEL_NN_FOLDER_NAME, params)
        for start, end in [(35000, 43000), (50000, 62000) , (507332, 568639)]:
            _, y_pred = nn.predict(start=start, end=end, write=False)

            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(cols_pred, cols_range)}).drop(columns=cols_range)
            tiles_df = Data.merge_dfs(inputs_outputs[start:end][cols_range_raw + ['datetime', 'SOLAR']], y_pred)
            Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                              show=False, smoothing_key='pred')
            for col in cols_range_raw:
                Plotter().plot_tile(tiles_df, det_rng=col, smoothing_key = 'pred')
            Plotter().plot_pred_true(tiles_df, cols_pred, cols_range_raw)
            Plotter.save(MODEL_NN_FOLDER_NAME, params, (start, end))
        if history.history['loss'][-1] < 0.005:
            get_feature_importance(nn.model_path, inputs_outputs, cols_range, cols_selected, num_sample=100, show=False)

def run_multimean_knn(inputs_outputs, cols_range, cols_selected):
    '''Runs the multi mean knn model'''
    multi_reg = MultiMeanKNeighborsRegressor(inputs_outputs, cols_range, cols_selected)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 63000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=cols_range)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, det_rng='Xpos')
    Plotter().show()
    Plotter.save(MODEL_NN_FOLDER_NAME)

def run_multimedian_knn(inputs_outputs, cols_range, cols_selected):
    '''Runs the multi median knn model'''
    multi_reg = MultiMedianKNeighborsRegressor(inputs_outputs, cols_range, cols_selected)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 73000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=cols_range)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, det_rng='Xpos')
    Plotter().show()
    Plotter.save(MODEL_NN_FOLDER_NAME)

def run_median(inputs_outputs, cols_range):
    '''Runs the multi median knn model'''

    mean_top = inputs_outputs['top'].mean()
    std_top = inputs_outputs['top'].std()
    for col in y_cols:
        mean_col = inputs_outputs[col].mean()
        std_col = inputs_outputs[col].std()

        inputs_outputs[f'{col}_orig'] = inputs_outputs[col]
        inputs_outputs[col] = (inputs_outputs[col] - mean_top) / std_top * std_col + mean_col
        inputs_outputs[col] = inputs_outputs[col].rolling(window=100).mean()

    median = np.median(inputs_outputs[['top', 'Xneg', 'Ypos', 'Yneg']], axis=1)
    y_median = pd.DataFrame(median, columns=['median'])
    y_median = y_median.rolling(window=50).mean()
    tiles_df = pd.concat([inputs_outputs[y_cols + ['datetime', 'SOLAR']], y_median], axis=1)
    mean_median = tiles_df['median'].mean()
    std_median = tiles_df['median'].std()
    for col in y_cols:
        mean_col = tiles_df[col].mean()
        std_col = tiles_df[col].std()
        tiles_df[f'{col}_smooth'] = (tiles_df['median'] - mean_median) / std_median * std_col + mean_col
        tiles_df[f'{col}_dif'] = inputs_outputs[f'{col}_orig'] - tiles_df['median']
    tiles_df.drop(columns=['median'], inplace=True)
    Plotter(df = tiles_df, label = 'median').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')
    Plotter().show()

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_dfs_from_pk_folder()
    # inputs_outputs_df = inputs_outputs_df.dropna().reset_index(drop=True)
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-06 05:30:22',
    #                                               stop='2023-12-06 09:15:28')

    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')
    x_cols = [col for col in x_cols if col not in x_cols_excluded]

    run_nn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_multimean_knn(inputs_outputs_df, y_cols, x_cols)
    # run_multimedian_knn(inputs_outputs_df, y_cols, x_cols)
    # run_median(inputs_outputs_df, y_cols)