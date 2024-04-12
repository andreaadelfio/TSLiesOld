"""Plotter module for plotting data points and curves."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import Logger, logger_decorator

class Plotter:
    """
    This class provides methods for plotting data points and curves.
    """
    logger = Logger('Plotter').get_logger()

    @logger_decorator(logger)
    def __init__(self, x = None, y = None, df: pd.DataFrame = None, xy: dict = None, label = ''):
        """
        Initialize the Plotter object.

        Parameters:
        - x (list): The x-coordinates of the data points (default: None).
        - y (list): The y-coordinates of the data points (default: None).
        - xy (dict): A dictionary of x, y, and smooth y values for multiple curves (default: None).
        - label (str): The label for the plot (default: '').
        """
        self.x = x
        self.y = y
        self.df = df
        self.xy = xy
        self.label = label

    @logger_decorator(logger)
    def plot(self, marker = '-', lw = 0, show = True):
        """
        Plot a single curve.

        Parameters:
        - marker (str): The marker style for the plot (default: '-').
        """
        plt.figure()
        plt.tight_layout(pad = 0.4)
        plt.plot(self.x, self.y, marker = marker, lw = lw, label = self.label)
        plt.legend()
        plt.title(self.label)
        plt.xlim(self.x[0], self.x[len(self.x) - 1])
        if show:
            plt.show()

    @logger_decorator(logger)
    def plot_tiles(self, marker = '-', lw = 0.2, with_smooth = False, show = True):
        """
        Plot multiple curves as tiles.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        i = 0
        _, axs = plt.subplots(len(self.xy), 1, sharex=True)
        plt.tight_layout(pad = 0.4)
        axs[0].set_title(self.label)
        for label, xy in self.xy.items():
            axs[i].plot(xy[0], xy[1], marker = marker, lw = lw, label=label)
            if with_smooth:
                axs[i].plot(xy[0], xy[2], marker = marker, label=f'{label} smooth')
            axs[i].legend()
            axs[i].grid()
            axs[i].set_xlim(xy[0][0], xy[0][-1])
            i += 1
        if show:
            plt.show()

    @logger_decorator(logger)
    def multiplot(self, marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plots multiple curves on the same figure.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        plt.figure()
        plt.tight_layout(pad = 0.4)
        plt.title(self.label)
        for label, xy in self.xy.items():
            plt.plot(xy[0], xy[1], marker = marker, lw = lw, label = label)
            if with_smooth:
                plt.plot(xy[0], xy[2], marker = marker, label = f'{label} smooth')
            plt.legend()
        plt.xlim(xy[0][0], xy[0][-1])
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_plot_tiles(self, x_col, excluded_cols = None, marker = '-', lw = 0.1, smoothing_key = 'smooth', show = True):
        """
        Plot multiple curves as tiles.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        fig.subplots_adjust(bottom = 0.06, hspace = 0)
        x = self.df[x_col]
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                axs[i].plot(x, self.df[column], marker = marker, lw = lw, label=column)
                if smoothing_key != '' and column + f'_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x, self.df[column + f'_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
                axs[i].legend(loc='upper right')
                axs[i].grid()
                axs[i].set_xlim(x[0], x[len(x) - 1])
                axs[i].tick_params(axis="x", labelrotation=30)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(x, self.df[column], marker = marker, lw = lw, label=column)
            axs.legend(loc='upper right')
            axs.grid()
            axs.set_xlim(x[0], x[len(x) - 1])
            axs.tick_params(axis="x", labelrotation=45)
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_multiplot(self, x_col, marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plots multiple curves on the same figure.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        plt.figure(self.label)
        plt.tight_layout(pad = 0.4)
        plt.title(self.label)
        x = self.df[x_col]
        for column in self.df.columns:
            if column != x_col:
                plt.plot(x, self.df[column], marker = marker, lw = lw, label = column)
                if with_smooth:
                    plt.plot(x, self.df[column + '_smooth'], marker = marker, label = f'{column} smooth')
                plt.legend()
        plt.xlim(x[0], x[len(x) - 1])
        if show:
            plt.show()


    @logger_decorator(logger)
    def plot_tile(self, tiles_df, det_rng='top', smoothing_key = 'smooth'):
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 12), num=det_rng, tight_layout=True)
            fig.subplots_adjust(hspace=0)

            axs[0].plot(pd.to_datetime(tiles_df['datetime']), tiles_df[det_rng], 'k-.')
            axs[0].plot(pd.to_datetime(tiles_df['datetime']), tiles_df[f'{det_rng}_{smoothing_key}'], 'r-')
            axs[0].set_title('foreground and background')
            axs[0].set_ylabel('Count Rate')

            axs[1].plot(pd.to_datetime(tiles_df['datetime']), tiles_df[det_rng] - tiles_df[f'{det_rng}_{smoothing_key}'], 'k-.')
            axs[1].plot(pd.to_datetime(tiles_df['datetime']).ffill(), [0 for _ in tiles_df['datetime'].ffill()], 'k-')
            axs[1].set_xlabel('time (YYYY-MM-DD hh:mm:ss)')
            plt.xticks(rotation=0)
            axs[1].set_ylabel('Residuals')

    @logger_decorator(logger)
    def plot_pred_true(self, tiles_df, col_range=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']):
        with sns.plotting_context("talk"):
            col_range_prev = [col.split('_')[0] for col in col_range]
            fig = plt.figure("pred_vs_true", layout="tight")
            fig.set_size_inches(24, 12)
            plt.axis('equal')
            plt.plot(tiles_df[col_range_prev], tiles_df[col_range], '.', alpha=0.2)
            min_y, max_y = min(tiles_df[col_range].min()), max(tiles_df[col_range].max())
            plt.plot([min_y, max_y], [min_y, max_y], '-')
            plt.xlabel('True signal')
            plt.ylabel('Predicted signal')
        plt.legend(col_range_prev)  

    @logger_decorator(logger)
    def plot_history(self, history, feature):
        plt.figure(f"history_{feature}", layout="tight")
        plt.plot(history.history[feature])
        plt.plot(history.history[f'val_{feature}'])
        plt.ylabel(feature)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    @logger_decorator(logger)
    def show_models_params(self, folder_name, features = {'top': 0.0004, 'Xpos': 0.0007, 'Xneg': 0.0007, 'Ypos': 0.0007, 'Yneg': 0.0007}):
        df = pd.read_csv(f'{folder_name}/models_params.csv', sep="\t").reset_index()
        df = df.assign(**{df.columns.tolist()[i+1]: df[df.columns.tolist()[i]] for i in range(len(df.columns.tolist()) - 1)}).drop(columns = 'index')
        for feature, value in features.items():
            df = df[df[feature] < value]
        print(df)
        
    @logger_decorator(logger)
    def show(self):
        plt.show()

    @logger_decorator(logger)
    def save(folder_name = '.', params = None, indexes = None):
        folder_name = f'{folder_name}/{params["model_id"]}' if params else folder_name
        for i in plt.get_fignums():
            title = plt.figure(i, figsize=(20, 12)).get_label()
            name = f'{title}.png' if not indexes else f'{title}_{indexes[0]}_{indexes[1]}.png'
            plt.savefig(f'{folder_name}/{name}')
        plt.close('all')

if __name__ == '__main__':
    Plotter().show_models_params('data/model_nn', features = {'epochs': 71})