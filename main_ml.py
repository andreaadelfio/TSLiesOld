"""Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 22/05/2024
TODO:
"""
import itertools
import pandas as pd
from scripts.plotter import Plotter
from scripts.utils import Data, File
from scripts.config import MODEL_NN_FOLDER_NAME
from scripts.nn import NN, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor, get_feature_importance

def run_nn(inputs_outputs, cols_range, cols_range_raw, cols_pred, cols_selected):
    '''Runs the neural network model'''
    Plotter().plot_correlation_matrix(inputs_outputs, show=False, save=True)

    nn = NN(inputs_outputs, cols_range, cols_selected)
    units_1_values = [90]
    units_2_values = [90]
    units_3_values = [30, 70]
    epochs_values = [20]
    bs_values = [1000]
    do_values = [0.02]
    norm_values = [0]
    drop_values = [0]
    opt_name_values = ['Adam']
    lr_values = [0.00009]
    loss_type_values = ['mae']

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
        for start, end in [(0, 73000)]:#(60000, 73000), (73000, -1)]:
            _, y_pred = nn.predict(start=start, end=end)
            # Plotter().plot_confusion_matric(inputs_outputs[start:end], y_pred, show=False)

            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(cols_pred, cols_range)}).drop(columns=cols_range)
            tiles_df = Data.merge_dfs(inputs_outputs[start:end][cols_range_raw + ['datetime', 'SOLAR']], y_pred)
            Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                              show=False, smoothing_key='pred')
            for col in cols_range_raw:
                Plotter().plot_tile(tiles_df, det_rng=col, smoothing_key = 'pred')
            print(tiles_df.columns.tolist())
            Plotter().plot_pred_true(tiles_df, cols_pred, cols_range_raw)
            Plotter.save(MODEL_NN_FOLDER_NAME, params, (start, end))

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

########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = File.read_dfs_from_pk_folder('/mnt/E28C2CB28C2C82E1/Users/Andrea/Documenti/inputs_outputs/pk')
    # inputs_outputs_df = pd.read_pickle('data/model_nn/0_/pk/bkg.pk')

    y_cols_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_smooth_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    y_pred_cols = [col + '_pred' for col in y_cols_raw]
    x_cols = [col for col in inputs_outputs_df.columns if col not in y_cols + y_smooth_cols + ['datetime']]

    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-06 05:30:22',
    #                                               stop='2023-12-06 09:15:28')
    inputs_outputs_df = inputs_outputs_df.dropna()
    Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col in x_cols], marker = ',', show = True, smoothing_key='smooth')

    run_nn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_multimean_knn(inputs_outputs_df, y_cols, x_cols)
    # run_multimedian_knn(inputs_outputs_df, y_cols, x_cols)

    MODEL_PATH = './data/model_nn/0/model.keras'
    get_feature_importance(MODEL_PATH, inputs_outputs_df, y_cols, x_cols, show=False)
