''' This script runs the model and plots the results. '''
from scripts.nn import NN
from scripts.utils import Data, File
from scripts.plotter import Plotter
from scripts.trigger import trigger

def run_trigger(inputs_outputs, y_cols, y_cols_raw, y_cols_pred, x_cols):
    '''Runs the model'''
    nn = NN(inputs_outputs, y_cols, x_cols)
    nn.set_scaler(inputs_outputs[x_cols])
    nn.set_model(model_path='data/model_nn/1/model.keras')
    start, end = 0, -1
    _, y_pred = nn.predict(start, end)

    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs[start:end][y_cols_raw + ['datetime', 'SOLAR']], y_pred)
    # Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
    #                                                   show=False, smoothing_key='pred')
    # for col in y_cols_raw:
    #     Plotter().plot_tile(tiles_df, det_rng=col, smoothing_key = 'pred')
    # Plotter.show()

    trigger(tiles_df, y_cols, y_cols_pred, threshold=5)


if __name__ == '__main__':
    inputs_outputs_df = File.read_dfs_from_pk_folder('/mnt/E28C2CB28C2C82E1/Users/Andrea/Documenti/inputs_outputs/pk')
    inputs_outputs_df = inputs_outputs_df.dropna()

    y_cols_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_smooth_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    y_pred_cols = [col + '_pred' for col in y_cols_raw]
    x_cols = [col for col in inputs_outputs_df.columns if col not in y_cols + y_smooth_cols + ['datetime']]

    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-06 05:30:22',
    #                                               stop='2023-12-06 09:15:28')
    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col in x_cols], marker = ',', show = True, smoothing_key='smooth')

    run_trigger(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)