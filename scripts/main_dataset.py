'''
Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 27/06/2024
TODO:
'''

import sys
import os
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.spacecraft import SpacecraftOpener
from modules.catalog import CatalogReader
from modules.plotter import Plotter
from modules.solar import SunMonitor
from modules.utils import Data, File, Time, Logger, logger_decorator
from modules.config import INPUTS_OUTPUTS_FILE_PATH

from scripts.main_config import h_names, y_cols, y_cols_raw, units, x_cols

logger = Logger('Main Dataset').get_logger()


@logger_decorator(logger)
def get_tiles_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(h_names=h_names, start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_catalog()
    # Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
    #                                                                 excluded_cols = ['MET'],
    #                                                                 marker = ',',
    #                                                                 smoothing_key='smooth',
    #                                                                 show = True)
    # print(tile_signal_df.head())
    # print(tile_signal_df.tail())
    # cr = CatalogReader(h_names=h_names, data_dir='data/LAT_ACD/Feb24-Jul24 output runs', start=0, end=-1)
    # tile_signal_df1 = cr.get_signal_df_from_catalog()
    # tile_signal_df1 = Data.get_masked_dataframe(data=tile_signal_df1, 
    #                                             start=tile_signal_df['datetime'].iloc[0],
    #                                             stop=tile_signal_df['datetime'].iloc[-1],
    #                                             reset_index=True)
    # tile_signal_df = tile_signal_df.merge(tile_signal_df1, on='datetime', suffixes=('', '_df1')).reset_index(drop=True)
    # print(tile_signal_df.head())
    # print(tile_signal_df.tail())
    # for col in y_cols_raw:
    #     tile_signal_df[col] = tile_signal_df[col] - tile_signal_df[f'{col}_df1']
    # Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
    #                                                                 excluded_cols = ['MET'],
    #                                                                 marker = ',',
    #                                                                 smoothing_key='smooth',
    #                                                                 show = True)
    # for col in y_cols_raw:
    #     tile_signal_df[col] = tile_signal_df[col] - tile_signal_df1[col]
    # print(tile_signal_df.head())
    # print(tile_signal_df.tail())

    runs_times = cr.get_runs_times()
    weeks_list = Time.get_week_from_datetime(runs_times)
    print(' done')
    return tile_signal_df, weeks_list

@logger_decorator(logger)
def get_sc_params_df(week, saa_exit_time):
    '''Get the spacecraft parameters dataframe'''
    sco = SpacecraftOpener()
    file = sco.get_sc_lat_weekly(week)
    sco.open(file, excluded_columns=['IN_SAA'])
    sc_params_df = sco.get_dataframe()
    sc_params_df = sco.get_good_quality_data(sc_params_df)
    sc_params_df, saa_exit_time = sco.add_saa_passage(sc_params_df, saa_exit_time)
    sc_params_df = sco.add_sun_occultation(sc_params_df)
    return sc_params_df, saa_exit_time

@logger_decorator(logger)
def get_solar_signal_df(week):
    '''Get the solar signal dataframe from the GOES data'''
    tstart, tend = Time.get_datetime_from_week(week)
    sm = SunMonitor(tstart, tend)
    file_goes = sm.fetch_goes_data()
    solar_signal_df = sm.merge_goes_data(file_goes)
    return solar_signal_df

@logger_decorator(logger)
def get_inputs_outputs_df():
    '''Get the inputs and outputs dataframe'''
    tile_signal_df, weeks_list = get_tiles_signal_df()
    # tile_signal_df = Data.get_masked_dataframe(data=tile_signal_df,
    #                                               start='2024-05-05 04:00:00',
    #                                               stop='2024-05-06 04:00:00')
    Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
                                                                    excluded_cols = ['MET'],
                                                                    marker = ',',
                                                                    smoothing_key='smooth',
                                                                    show = True)
    saa_exit_time = 0
    inputs_outputs_list = []
    for week in tqdm([week for week in weeks_list if week not in ()], desc='Creating weekly datasets'):
        sc_params_df, saa_exit_time = get_sc_params_df(week, saa_exit_time)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        solar_signal_df = get_solar_signal_df(week)
        inputs_outputs = Data.merge_dfs(inputs_outputs, solar_signal_df)
        inputs_outputs['GOES_XRSA_HARD_EARTH_OCCULTED'] = (1 - inputs_outputs['SUN_IS_EARTH_OCCULTED']) * inputs_outputs['GOES_XRSA_HARD']
        inputs_outputs['GOES_XRSB_SOFT_EARTH_OCCULTED'] = (1 - inputs_outputs['SUN_IS_EARTH_OCCULTED']) * inputs_outputs['GOES_XRSB_SOFT']

        # Plotter(df = inputs_outputs, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
        #                                                                 excluded_cols = [col for col in inputs_outputs.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']],
        #                                                                 marker = ',',
        #                                                                 smoothing_key='smooth',
        #                                                                 show = True)
        File.write_df_on_file(inputs_outputs,
                              filename=f'{INPUTS_OUTPUTS_FILE_PATH}_w{week}',
                              fmt='both')
        inputs_outputs_list.append(inputs_outputs)
    return pd.concat(inputs_outputs_list, ignore_index=True)


# MARK: Main
if __name__ == '__main__':
    # inputs_outputs_df = get_inputs_outputs_df()
    inputs_outputs_df = File().read_dfs_from_weekly_pk_folder(start=0, stop=1000)
    inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
                                                  start='2024-06-20 22:35:00',
                                                  stop='2024-06-20 23:40:00')

    Plotter(df = inputs_outputs_df[['GOES_XRSA_HARD', 'GOES_XRSB_SOFT', 'MET', 'datetime']], label = 'Outputs').df_plot_tiles(
                                                    y_cols=y_cols,
                                                    x_col ='MET',
                                                    top_x_col='datetime',
                                                    excluded_cols = y_cols_raw + ['SUN_IS_EARTH_OCCULTED', 'MET', 'datetime', 'TIME_FROM_SAA', 'SAA_EXIT', 'GOES_XRSA_HARD_EARTH_OCCULTED', 'GOES_XRSB_SOFT_EARTH_OCCULTED'],
                                                    init_marker=',',
                                                    smoothing_key='smooth',
                                                    save = True,
                                                    units=units)
    Plotter.show()
