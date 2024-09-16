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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.spacecraft import SpacecraftOpener
from modules.catalog import CatalogReader
from modules.plotter import Plotter
from modules.solar import SunMonitor
from modules.utils import Data, File, Time, Logger, logger_decorator
from modules.config import INPUTS_OUTPUTS_FILE_PATH

from scripts.main_config import h_names

logger = Logger('Main Dataset').get_logger()


@logger_decorator(logger)
def get_tiles_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(h_names=h_names, start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_catalog()
    tile_signal_df = cr.add_smoothing(tile_signal_df)
    runs_times = cr.get_runs_times()
    weeks_list = Time.get_week_from_datetime(runs_times)
    print(' done')
    return tile_signal_df, weeks_list

@logger_decorator(logger)
def get_sc_params_df(week, saa_exit_time):
    '''Get the spacecraft parameters dataframe'''
    print('Sc params...', end='')
    sco = SpacecraftOpener()
    file = sco.get_sc_lat_weekly(week)
    sco.open(file, excluded_columns=['IN_SAA'])
    sc_params_df = sco.get_dataframe()
    sc_params_df = sco.get_good_quality_data(sc_params_df)
    sc_params_df, saa_exit_time = sco.add_saa_passage(sc_params_df, saa_exit_time)
    sc_params_df = sco.add_sun_occultation(sc_params_df)
    print(' done')
    return sc_params_df, saa_exit_time

@logger_decorator(logger)
def get_solar_signal_df(week):
    '''Get the solar signal dataframe from the GOES data'''
    print('Solar...', end='')
    tstart, tend = Time.get_datetime_from_week(week)
    sm = SunMonitor(tstart, tend)
    file_goes = sm.fetch_goes_data()
    solar_signal_df = sm.merge_goes_data(file_goes)
    print(' done')
    return solar_signal_df

@logger_decorator(logger)
def get_inputs_outputs_df():
    '''Get the inputs and outputs dataframe'''
    tile_signal_df, weeks_list = get_tiles_signal_df()
    # tile_signal_df = Data.get_masked_dataframe(data=tile_signal_df,
    #                                               start='2024-01-05 04:00:00',
    #                                               stop='2024-01-06 04:00:00')
    # Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'MET',
    #                                                                 excluded_cols = ['MET'],
    #                                                                 marker = ',',
    #                                                                 smoothing_key='smooth',
    #                                                                 show = True)
    saa_exit_time = 0
    inputs_outputs_list = []
    for week in [week for week in weeks_list if week not in ()]:
        sc_params_df, saa_exit_time = get_sc_params_df(week, saa_exit_time)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        solar_signal_df = get_solar_signal_df(week)
        inputs_outputs = Data.merge_dfs(inputs_outputs, solar_signal_df)
        inputs_outputs['SUN_IS_OCCULTED'] = (1 - inputs_outputs['SUN_IS_OCCULTED']) * inputs_outputs['SOLAR']

        # Plotter(df = inputs_outputs, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
        #                                                                 excluded_cols = [col for col in inputs_outputs.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']],
        #                                                                 marker = ',',
        #                                                                 smoothing_key='smooth',
        #                                                                 show = True)
        File.write_df_on_file(inputs_outputs,
                              filename=f'{INPUTS_OUTPUTS_FILE_PATH}_w{week}',
                              fmt='both')
        inputs_outputs_list.append(inputs_outputs)
    print(' done')
    return pd.concat(inputs_outputs_list, ignore_index=True)


# MARK: Main
if __name__ == '__main__':
    inputs_outputs_df = get_inputs_outputs_df()
    # inputs_outputs_df = File.read_dfs_from_pk_folder()
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-21 21:50:00',
    #                                               stop='2023-12-31 22:08:00')



    Plotter(df = inputs_outputs_df[['Xpos', 'top', 'Xpos_smooth', 'top_smooth', 'datetime', 'SOLAR']], label = 'Outputs').df_plot_tiles(x_col = 'datetime',
                                                                     excluded_cols = [],
                                                                     marker = ',', smoothing_key='smooth',
                                                                     show = False)
    Plotter.show()
