"""Main file to run the project
Author: Andrea Adelfio
Created date: 17/05/2024
Modified date: 17/05/2024
TODO:
"""
import itertools
import pandas as pd
from scripts.plotter import Plotter
from scripts.utils import Data, File
from scripts.config import MODEL_NN_FOLDER_NAME
from scripts.nn import NN, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor

def run_nn(inputs_outputs):
    '''Runs the neural network model'''
    col_range_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    col_range = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    # col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    # col_range = ['Xpos']
    col_pred = [col + '_pred' for col in col_range_raw]
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
                    'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 'GEOMAG_LAT',
                    'LAMBDA', 'RA_SCZ', 'DEC_SCZ', 'RA_SCX', 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE',
                    'ROCK_ANGLE', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
                    'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2', 'SOLAR']


    nn = NN(inputs_outputs, col_range, col_selected)
    units_1_values = [90]
    units_2_values = [90]
    units_3_values = [30, 70]
    epochs_values = [6]
    bs_values = [1000]
    do_values = [0.02]
    norm_values = [0]
    drop_values = [0]
    opt_name_values = ['Adam']
    lr_values = [0.00009]
    loss_type_values = ['mae']

    hyperparams_combinations = list(itertools.product(units_1_values, units_2_values, units_3_values, norm_values, drop_values, epochs_values, bs_values, do_values, opt_name_values, lr_values, loss_type_values))
    hyperparams_combinations = nn.trim_hyperparams_combinations(hyperparams_combinations)
    # hyperparams_combinations = nn.use_previous_hyperparams_combinations(hyperparams_combinations)
        
    for model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
        params = {'model_id': model_id, 'units_1': units_1, 'units_2': units_2, 'units_3': units_3, 'norm': norm, 'drop': drop, 'epochs': epochs, 'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
        nn.set_hyperparams(params)
        nn.create_model()
        history = nn.train()
        Plotter().plot_history(history, 'loss')
        Plotter().plot_history(history, 'accuracy')
        nn.update_summary()
        Plotter.save(MODEL_NN_FOLDER_NAME, params)
        for start, end in [(60000, 73000)]:#, (73000, -1)]:
            _, y_pred = nn.predict(start=start, end=end)

            y_pred = y_pred.assign(**{col: y_pred[col_init] for col, col_init in zip(col_pred, col_range)}).drop(columns=col_range)
            tiles_df = Data.merge_dfs(inputs_outputs[start:end][col_range_raw + ['datetime', 'SOLAR']], y_pred)
            Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',', show=False, smoothing_key='pred')
            for col in col_range_raw:
                Plotter().plot_tile(tiles_df, det_rng=col, smoothing_key = 'pred')
            print(tiles_df.columns.tolist())
            Plotter().plot_pred_true(tiles_df, col_pred, col_range_raw)
            Plotter.save(MODEL_NN_FOLDER_NAME, params, (start, end))

def run_multimean_knn(inputs_outputs):
    '''Runs the multi mean knn model'''
    col_range = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    # col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
                    'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 'GEOMAG_LAT',
                    'LAMBDA', 'RA_SCZ', 'DEC_SCZ', 'RA_SCX', 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE',
                    'ROCK_ANGLE', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
                    'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2', 'SOLAR']

    multi_reg = MultiMeanKNeighborsRegressor(inputs_outputs, col_range, col_selected)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 63000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=col_range)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, det_rng='Xpos')
    Plotter().show()
    Plotter.save(MODEL_NN_FOLDER_NAME)

def run_multimedian_knn(inputs_outputs):
    '''Runs the multi median knn model'''
    col_range = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    # col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
                    'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 'GEOMAG_LAT',
                    'LAMBDA', 'RA_SCZ', 'DEC_SCZ', 'RA_SCX', 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE',
                    'ROCK_ANGLE', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
                    'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2', 'SOLAR']

    multi_reg = MultiMedianKNeighborsRegressor(inputs_outputs, col_range, col_selected)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 63000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=col_range)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, det_rng='Xpos')
    Plotter().show()
    Plotter.save(MODEL_NN_FOLDER_NAME)

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_df_from_folder()
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df, start='2023-12-07 04:00:00', stop='2023-12-08 04:00:00')
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')

    run_nn(inputs_outputs_df)
    # run_multimean_knn(inputs_outputs_df)
    # run_multimedian_knn(inputs_outputs_df)
