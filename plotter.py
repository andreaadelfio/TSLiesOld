"""Plotter module for plotting data points and curves."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Plotter:
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
        if show: plt.show()

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
        if show: plt.show()

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
        if show: plt.show()

    def df_plot_tiles(self, x_col, excluded_cols = [], marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plot multiple curves as tiles.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        df_columns = [column for column in self.df.columns if '_smooth' not in column and column not in excluded_cols and column != x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        _, axs = plt.subplots(n_rows, n_cols, sharex=True, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        x = self.df[x_col]
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                axs[i].plot(x, self.df[column], marker = marker, lw = lw, label=column)
                if with_smooth and column + '_smooth' in self.df.columns:
                    axs[i].plot(x, self.df[column + '_smooth'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} smooth')
                axs[i].legend()
                axs[i].grid()
                axs[i].set_xlim(x[0], x[len(x) - 1])
            for j in range(i, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(x, self.df[column], marker = marker, lw = lw, label=column)
            axs.legend()
            axs.grid()
            axs.set_xlim(x[0], x[len(x) - 1])
        if show: plt.show()

    def df_multiplot(self, x_col, marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plots multiple curves on the same figure.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        plt.figure()
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
        if show: plt.show()

    def show():
        plt.show()