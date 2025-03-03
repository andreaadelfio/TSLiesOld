'''
Plotter module for plotting data points and curves.
'''
import os
import operator
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
try:
    from modules.utils import Logger, logger_decorator, Time
    from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME
except:
    from utils import Logger, logger_decorator, Time
    from config import BACKGROUND_PREDICTION_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME


class Plotter:
    '''
    This class provides methods for plotting data points and curves.
    '''
    logger = Logger('Plotter').get_logger()
    
    @logger_decorator(logger)
    def __init__(self, x = None, y = None, df: pd.DataFrame = None, xy: dict = None, label = '', latex = False):
        '''
        Initialize the Plotter object.

        Parameters:
        ----------
            x (list): The x-coordinates of the data points (default: None).
            y (list): The y-coordinates of the data points (default: None).
            df (pd.DataFrame): The y-coordinates of the data points (default: None).
            xy (dict): A dictionary of x, y, and smooth y values for multiple curves (default: None).
            label (str): The label for the plot (default: '').
        '''
        self.x = x
        self.y = y
        self.df = df
        self.xy = xy
        self.label = label
        if latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans.serif",
                "font.sans-serif": ["Helvetica"],
                "font.size": 25
            })

    @logger_decorator(logger)
    def plot(self, marker = '-', lw = 0, show = True):
        '''
        Plot a single curve.

        Parameters:
        ----------
            marker (str): The marker style for the plot (default: '-').
        '''
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
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
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
        '''
        Plots multiple curves on the same figure. Uses the `xy` variable passed to the Plotter.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
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
    def df_plot_corr_tiles(self, x_col, excluded_cols = None, marker = '-', ms = 1, lw = 0.1, smoothing_key = 'smooth', show = True):
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != 'datetime']
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        fig.subplots_adjust(bottom = 0.08, hspace = 0.13, wspace = 0.08, left = 0.04)
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                if column == x_col:
                    x = self.df['datetime']
                    axs[i].plot(x, self.df[column], marker = marker, ms = ms, lw = lw, label=column)
                else:
                    x = self.df[x_col]
                    axs[i].plot(self.df[column], x, marker = marker, ms = ms, lw = lw, label=column)
                if smoothing_key != '' and f'{column}_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x, self.df[f'{column}_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
                axs[i].legend(loc='upper right')
                axs[i].set_xlabel(column)
                axs[i].set_ylabel(x_col)
                axs[i].grid()
                axs[i].tick_params(axis="x", labelrotation=30)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(self.df[column], x, marker = marker, lw = lw, label=column)
            axs.set_xlabel(column)
            axs.set_ylabel(x_col)
            axs.legend(loc='upper right')
            axs.grid()
            axs.tick_params(axis="x", labelrotation=45)
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_plot_tiles(self, y_cols, x_col, top_x_col = None, excluded_cols = None, init_marker = '-', lw = 0.1, smoothing_key = 'smooth', show = True, save = False, units = {}):
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''

        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and '_std' not in column and column not in excluded_cols and column != x_col and column != top_x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig = plt.figure(figsize=(12, 6), num=self.label)
        x = self.df[x_col]
        axs = [fig.add_subplot(n_rows, n_cols, 1)]
        for i in range(1, n_plots):
            axs.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axs[0]))

        for i, column in tqdm(enumerate(df_columns), desc='Plotting tiles'):
            fmt = init_marker
            plot_lw = lw
            if column in y_cols:
                fmt = 'k-.'
                plot_lw = None
            axs[i].plot(x.values, self.df[column], fmt, lw=plot_lw, label=f"{column} [{units[column] if column in units else 'NA'}]")

            if smoothing_key and f'{column}_{smoothing_key}' in self.df.columns:
                axs[i].plot(x.values, self.df[f'{column}_{smoothing_key}'], 'r-', ms=0.2, label=f'{column} {smoothing_key}')
                if f'{column}_std' in self.df.columns:
                    axs[i].fill_between(x.values, self.df[f'{column}_{smoothing_key}']-self.df[f'{column}_std'], self.df[f'{column}_{smoothing_key}']+self.df[f'{column}_std'], label='prediction error')
            
            if i >= n_cols * (n_rows - 1) - (n_cols * n_rows - n_plots):
                axs[i].tick_params(axis="x", labelrotation=30)
                plt.setp(axs[i].get_xticklabels(), visible=True)
                offset_text = axs[i].get_xaxis().get_offset_text().get_text()
                xlabel = f'{x_col} ({offset_text})' if offset_text else f'{x_col} ({x.iloc[0]})'
                axs[i].set_xlabel(xlabel)
            else:
                plt.setp(axs[i].get_xticklabels(), visible=False)
            axs[i].get_xaxis().get_offset_text().set_visible(False)
            
            axs[i].legend(loc='upper right', fontsize=13)
            axs[i].grid()
            axs[i].set_xlim(x.iloc[0], x.iloc[-1])
            axs[i].tick_params(axis="y", labelrotation=30)
            if top_x_col and top_x_col in self.df.columns and i < n_cols:
                if top_x_col == 'datetime':
                    secax = axs[i].secondary_xaxis('top', functions=(Time.from_met_to_datetime, lambda x: x))
                    secax.set_xlabel(f'{top_x_col} ({self.df[top_x_col].iloc[0]})')
                elif top_x_col == 'MET':
                    secax = axs[i].secondary_xaxis('top', functions=(Time.date2yday, lambda x: x))
                    secax.get_xaxis().get_offset_text().set_visible(False)
                    secax.set_xlabel(f'{top_x_col} ({self.df[top_x_col].iloc[0] / 1e8} $\dot 10^8$)')
                secax.tick_params(axis="x", labelrotation=30)
                # secax.set_xticklabels(labels=secax.xaxis.get_majorticklabels(), ha='left')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        fig.subplots_adjust(hspace=0, right=0.97, left=0.05)
        # plt.tight_layout()
        if show:
            plt.show()

        if save:
            fig.savefig(
                os.path.join(self.label),
                dpi=200, bbox_inches='tight')
            plt.close(fig)

    @logger_decorator(logger)
    def plot_tile(self, tiles_df, face='top', col='datetime', smoothing_key = 'smooth', units = {}, support_vars = None, show_std=True, save=False):
        # with sns.plotting_context("talk"):
        fig, axs = plt.subplots(2 + len(support_vars), 1, sharex=True, figsize=(20, 12), num=face)
        # fig.subplots_adjust(hspace=0)
        axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[face], 'k-.', label=f'{face}')
        axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[f'{face}_{smoothing_key}'], 'r-', label='background')
        axs[0].set_ylabel(f'{units[face] if face in units else "NA"}')

        axs[1].plot(pd.to_datetime(tiles_df[col]), tiles_df[face] - tiles_df[f'{face}_{smoothing_key}'], 'k-.', label='residuals')

        if f'{face}_std' in tiles_df.columns and show_std:
            axs[0].fill_between(pd.to_datetime(tiles_df[col]), tiles_df[f'{face}_{smoothing_key}']-tiles_df[f'{face}_std'], tiles_df[f'{face}_{smoothing_key}']+tiles_df[f'{face}_std'], label='prediction error')
            axs[1].fill_between(pd.to_datetime(tiles_df[col]), -tiles_df[f'{face}_std'], tiles_df[f'{face}_std'])
        
        axs[1].axhline(0, color='red')
        # axs[1].plot(pd.to_datetime(tiles_df[col]).ffill(), [0 for _ in tiles_df[col].ffill()], 'k-')
        axs[1].set_xlim(tiles_df[col].iloc[0], tiles_df[col].iloc[-1])
        plt.xticks(rotation=0)
        axs[1].set_ylabel(f'{units[face] if face in units else "NA"}')
        axs[0].legend()
        axs[1].legend()

        for i in range(len(support_vars)):
            axs[2 + i].plot(pd.to_datetime(tiles_df[col]), tiles_df[support_vars[i]], color='green', label=f'{support_vars[i]}')
            axs[2 + i].set_xlabel(f"{col} ({tiles_df[col].iloc[0]})")
            axs[2 + i].set_ylabel(f"{units[support_vars[i]] if support_vars[i] in units else 'NA'}")
            axs[2 + i].legend()

        plt.show()
        if save:
            print(os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'Output.png'))
            fig.savefig(
                os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'Output.png'),
                dpi=200, bbox_inches='tight')

    @logger_decorator(logger)
    def plot_pred_true(self, tiles_df, col_pred=['top_pred', 'Xpos_pred', 'Xneg_pred', 'Ypos_pred', 'Yneg_pred'], y_cols_raw=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']):
        with sns.plotting_context("talk"):
            fig = plt.figure("pred_vs_true", layout="tight")
            fig.set_size_inches(24, 12)
            plt.axis('equal')
            plt.plot(tiles_df[y_cols_raw], tiles_df[col_pred], '.', alpha=0.2)
            min_y, max_y = min(tiles_df[col_pred].min()), max(tiles_df[col_pred].max())
            plt.plot([min_y, max_y], [min_y, max_y], '-')
            plt.xlabel('True signal')
            plt.ylabel('Predicted signal')
        plt.legend(y_cols_raw)

    @logger_decorator(logger)
    def plot_history(self, history):
        for feature in history.history.keys():
            if 'val' in feature or feature == 'lr':
                continue
            plt.figure(f"history_{feature}", layout="tight")
            plt.plot(history.history[feature][1:])
            plt.plot(history.history[f'val_{feature}'][1:])
            plt.ylabel(feature)
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')

    @logger_decorator(logger)
    def plot_correlation_matrix(self, show = True, save = False):
        '''Function to plot the correlation matrix.'''
        correlations = self.df.corr()
        plt.figure(figsize=(18, 18), num='correlations_matrix')
        sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME)

    @logger_decorator(logger)
    def plot_confusion_matrix(self, y_true, y_pred, show = True):
        '''Function to plot the confusion matrix.'''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8), num='confusion_matrix')
        sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if show:
            plt.show()

    @logger_decorator(logger)
    def plot_anomalies(self, support_vars, thresholds, tiles_df, y_cols, y_pred_cols, save=True, show=False, extension='png', units={}):
        '''Plots the anomalies passed as `df` in Plotter.'''
        if not os.path.exists(PLOT_TRIGGER_FOLDER_NAME):
            os.makedirs(PLOT_TRIGGER_FOLDER_NAME)

        for anomaly_start, anomalies in tqdm(self.df.items(), desc='Plotting anomalies'):
            faces = list(anomalies.keys())
            
            # Define the number of rows dynamically (space between signal and residuals + support vars)
            num_signal_residual_pairs = len(y_cols)
            num_support_vars = len(support_vars)
            total_rows = 2 * num_signal_residual_pairs + num_support_vars

            # Use GridSpec for custom subplot layout
            fig = plt.figure(figsize=(8.5, 7.5 + 0.37 * total_rows))
            gs0 = GridSpec(6, 1, figure=fig, hspace=0.2, left=0.1, right=0.99, top=0.98, bottom=0.04)
            gss = []
            for i in range(num_signal_residual_pairs):
                gss.append(GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i], hspace=0.))
            gss.append(GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1], hspace=0.))

            axs = []

            anomaly_end = -1
            for anomaly in anomalies.values():
                if anomaly['stopping_time'] > anomaly_end:
                    anomaly_end = anomaly['stopping_time']
            anomaly_delta = max((anomaly_end - int(anomaly_start)) // 5, 120)
            start = max(int(anomaly_start) - anomaly_delta, 0)
            end = min(anomaly_end + anomaly_delta, len(tiles_df))
            start_xlim = tiles_df['datetime'][int(anomaly_start)] - timedelta(seconds=anomaly_delta)
            end_xlim = tiles_df['datetime'][anomaly_end] + timedelta(seconds=anomaly_delta-1)
            i=0
            for face, face_pred in zip(y_cols, y_pred_cols):
                current_row = 0
                ax_signal = fig.add_subplot(gss[i][:-1, :])
                current_row += 1
                axs.append(ax_signal)

                ax_residuals = fig.add_subplot(gss[i][-1, :], sharex=ax_signal)
                current_row += 1
                axs.append(ax_residuals)

                current_row += 1

                face_color = "black"
                if face in faces:
                    face_color = None
                    changepoint = anomalies[face]['changepoint']
                    stopping_time = anomalies[face]['stopping_time']
                    ax_signal.axvline(tiles_df['datetime'][changepoint], color='red', lw=0.8)
                    ax_signal.axvline(tiles_df['datetime'][stopping_time], color='red', lw=0.8)
                    ax_signal.fill(
                        (tiles_df['datetime'][int(anomaly_start)], tiles_df['datetime'][anomaly_end],
                        tiles_df['datetime'][anomaly_end], tiles_df['datetime'][int(anomaly_start)]),
                        (-5, -5, 15, 15), color="yellow", alpha=0.1
                    )
                    ax_residuals.axvline(tiles_df['datetime'][changepoint], color='red', lw=0.8)
                    ax_residuals.axvline(tiles_df['datetime'][stopping_time], color='red', lw=0.8)
                    ax_residuals.fill(
                        (tiles_df['datetime'][int(anomaly_start)], tiles_df['datetime'][anomaly_end],
                        tiles_df['datetime'][anomaly_end], tiles_df['datetime'][int(anomaly_start)]),
                        (-5, -5, 15, 15), color="yellow", alpha=0.1
                    )
                ax_residuals.set_ylabel(f'{face} [{units[face]}]')
                ax_residuals.yaxis.set_label_coords(-0.06,1.02)

                # ax_residuals.set_ylabel()
                ax_residuals.axhline(thresholds[face], color="orange", label=f"${thresholds[face]}\\sigma$")
                ax_residuals.plot(tiles_df[start:end]['datetime'], tiles_df[start:end][f'{face}_significance'],
                                color="blue", label=r"k-score $S$", lw=0.7)
                # ax_residuals.plot(tiles_df[start:end]['datetime'], tiles_df[start:end][face] - tiles_df[start:end][face_pred],
                #                 label='residuals', color=face_color)
                ax_signal.plot(tiles_df[start:end]['datetime'], tiles_df[start:end][face], label='signal',
                            color=face_color)
                ax_signal.plot(tiles_df[start:end]['datetime'], tiles_df[start:end][face_pred], label='background',
                            color='red')
                ax_signal.legend(loc="upper left")
                ax_residuals.legend(loc="upper left")
                max_val = max(max(tiles_df[start:end][f'{face}_significance']), thresholds[face] * 1.1)
                ax_signal.set_ylim(min(tiles_df[start:end][face]), 1.01 * max(tiles_df[start:end][face]))
                ax_residuals.set_ylim(0, 1.01 * max_val)
                ax_signal.set_xlim(start_xlim, end_xlim)
                ax_residuals.set_xlim(start_xlim, end_xlim)

                plt.setp(ax_signal.get_xticklabels(), visible=False)
                i+=1

            for var in support_vars:
                ax_support = fig.add_subplot(gss[i][0:1, :])
                axs.append(ax_support)

                ax_support.plot(tiles_df[start:end]['datetime'], tiles_df[start:end][var], color="green", label=var)
                ax_support.set_ylabel(f'[{units[var] if var in units else "NA"}]')
                ax_support.legend(loc="upper left")

            axs[-1].set_xlim(start_xlim, end_xlim)
            start_datetime = tiles_df['datetime'][int(anomaly_start)]
            stop_datetime = tiles_df['datetime'][anomaly_end]
            axs[-1].set_xlabel(f"datetime {start_datetime - timedelta(seconds=anomaly_delta)}")
            axs[0].set_title(f"Triggers in ${', '.join(faces)}$ between {start_datetime} and {stop_datetime}")

            if save:
                fig.savefig(
                    os.path.join(PLOT_TRIGGER_FOLDER_NAME, f"{tiles_df['datetime'][changepoint]}_{'_'.join(faces)}.{extension}"),
                    dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            plt.close('all')

    @logger_decorator(logger)
    @staticmethod
    def show():
        '''Shows the plots'''
        plt.show()

    @logger_decorator(logger)
    @staticmethod
    def save(folder_name = '.', params = None, indexes: tuple = None):
        '''Saves the plots'''
        folder_name = os.path.dirname(params['model_path']) if params else folder_name
        for i in plt.get_fignums():
            title = plt.figure(i).get_label()
            name = f'{title}.png' if not indexes else f'{title}_{indexes[0]}_{indexes[1]}.png'
            plt.savefig(os.path.join(folder_name, name) if not folder_name.endswith('png') else folder_name)
        plt.close('all')

if __name__ == '__main__':
    print('to do')