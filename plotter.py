"""Plotter module for plotting data points and curves."""
import matplotlib.pyplot as plt
import pandas as pd

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
        plt.title(self.label)
        for label, xy in self.xy.items():
            plt.plot(xy[0], xy[1], marker = marker, lw = lw, label = label)
            if with_smooth:
                plt.plot(xy[0], xy[2], marker = marker, label = f'{label} smooth')
            plt.legend()
        plt.xlim(xy[0][0], xy[0][-1])
        if show: plt.show()

    def df_plot_tiles(self, x_column, marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plot multiple curves as tiles.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        i = 0
        _, axs = plt.subplots(len(self.df.columns) - 1, 1, sharex=True)
        x = self.df[x_column]
        axs[0].set_title(self.label)
        for column in self.df.columns:
            if column != x_column:
                axs[i].plot(x, self.df[column], marker = marker, lw = lw, label=column)
                if with_smooth:
                    axs[i].plot(x, self.df[column + '_smooth'], marker = marker, label=f'{column} smooth')
                axs[i].legend()
                axs[i].grid()
                axs[i].set_xlim(x[0], x[len(x) - 1])
                i += 1
        if show: plt.show()

    def df_multiplot(self, x_column, marker = '-', lw = 0.1, with_smooth = False, show = True):
        """
        Plots multiple curves on the same figure.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        plt.figure()
        plt.title(self.label)
        x = self.df[x_column]
        for column in self.df.columns:
            if column != x_column:
                plt.plot(x, self.df[column], marker = marker, lw = lw, label = column)
                if with_smooth:
                    plt.plot(x, self.df[column + '_smooth'], marker = marker, label = f'{column} smooth')
                plt.legend()
        plt.xlim(x[0], x[len(x) - 1])
        if show: plt.show()

    def show():
        plt.show()