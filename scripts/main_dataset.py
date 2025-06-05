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

from scripts.main_config import h_names, y_smooth_cols, y_cols, y_cols_raw, units, x_cols, latex_y_cols

logger = Logger('Main Dataset').get_logger()


@logger_decorator(logger)
def get_tiles_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(h_names=h_names, start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_catalog()
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
    Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'MET',
                                                                    top_x_col='datetime',
                                                                    latex_y_cols=latex_y_cols,
                                                                    y_cols=y_cols,
                                                                    excluded_cols = x_cols + ['datetime', 'MET'],
                                                                    smoothing_key='smooth',
                                                                    show = True)
    saa_exit_time = 0
    for week in tqdm([week for week in weeks_list if week not in []], desc='Creating weekly datasets'):
        sc_params_df, saa_exit_time = get_sc_params_df(week, saa_exit_time)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        if len(inputs_outputs) == 0:
            continue
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
                              filename=os.path.join(INPUTS_OUTPUTS_FILE_PATH, f'w{week}'),
                              fmt='pk')

# MARK: Main
if __name__ == '__main__':
    # inputs_outputs_df = get_inputs_outputs_df()
    inputs_outputs_df = File().read_dfs_from_weekly_pk_folder(start=864, stop=1000, cols_list=y_cols + x_cols + y_smooth_cols + ['datetime'], y_cols=y_cols)
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2024-01-30 22:35:00',
    #                                               stop='2024-02-01 23:40:00')
    # for date in [('2024-08-25 00:00:00.827','2024-08-26 05:29:55.045')]:
    #     tmp_inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                                   start=date[0],
    #                                                   stop=date[1])
    #     if len(tmp_inputs_outputs_df) > 0:
    Plotter(df = inputs_outputs_df, label = 'Outputs').df_plot_tiles(
                                                        y_cols=y_cols,
                                                        latex_y_cols=latex_y_cols,
                                                        x_col ='MET',
                                                        top_x_col='datetime',
                                                        excluded_cols = x_cols + ['SUN_IS_EARTH_OCCULTED', 'MET', 'datetime', 'TIME_FROM_SAA', 'SAA_EXIT', 'GOES_XRSA_HARD_EARTH_OCCULTED', 'GOES_XRSB_SOFT_EARTH_OCCULTED'],
                                                        smoothing_key='smooth',
                                                        save = True,
                                                        units=units)
    
    Plotter.show()
