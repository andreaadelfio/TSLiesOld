'''Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 17/05/2024
TODO:
'''
import pandas as pd
from scripts.spacecraftopener import SpacecraftOpener
from scripts.catalogreader import CatalogReader
from scripts.plotter import Plotter
from scripts.sunmonitor import SunMonitor
from scripts.utils import Data, File, Time
from scripts.config import INPUTS_OUTPUTS_FILE_PATH


def get_good_data_quality(dataframe) -> pd.DataFrame:
    '''
    Returns the dataframe masked for good data quality.

    Returns:
        pandas.DataFrame: The dataframe containing the spacecraft data with good data quality.
    '''
    return dataframe[dataframe['DATA_QUAL'] != 0]

def get_tile_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(start=0, end=2)
    tile_signal_df = cr.get_signal_df_from_catalog()
    tile_signal_df = cr.add_smoothing(tile_signal_df)
    runs_times = cr.get_runs_times()
    weeks_list = Time.get_week_from_datetime(runs_times)
    print(' done')
    return tile_signal_df, weeks_list

def get_sc_params_df(week):
    '''Get the spacecraft parameters dataframe'''
    print('Sc params...', end='')
    sco = SpacecraftOpener()
    file = sco.get_sc_lat_weekly(week)
    sco.open(file)
    sc_params_df = sco.get_dataframe()
    print(' done')
    return sc_params_df

def get_solar_signal_df(week):
    '''Get the solar signal dataframe from the GOES data'''
    print('Solar...', end='')
    tstart, tend = Time.get_datetime_from_week(week)
    sm = SunMonitor(tstart, tend)
    file_goes = sm.fetch_goes_data()
    solar_signal_df = sm.merge_goes_data(file_goes)
    print(' done')
    return solar_signal_df

def get_inputs_outputs_df():
    '''Get the inputs and outputs dataframe'''
    tile_signal_df, weeks_list = get_tile_signal_df()
    for week in weeks_list:
        sc_params_df = get_sc_params_df(week)
        sc_params_df = get_good_data_quality(sc_params_df)
        solar_signal_df = get_solar_signal_df(week)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        inputs_outputs = Data.merge_dfs(inputs_outputs, solar_signal_df)
        File.write_df_on_file(inputs_outputs, filename=INPUTS_OUTPUTS_FILE_PATH + f'_w{week}')
        print(' done')
    return inputs_outputs


########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = get_inputs_outputs_df()
    inputs_outputs_df = File.read_df_from_folder()
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start='2023-12-07 04:00:00',
    #                                               stop='2023-12-08 04:00:00')
    # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')


    col_range_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
                    'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 
                    'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ', 'START', 'STOP', 'MET', 'IN_SAA',
                    'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'LIVETIME', 'DEC_SCZ', 'RA_SCX',
                    'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE', 'QSJ_1', 'QSJ_2',
                    'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN', 'SC_VELOCITY_0', 'SC_VELOCITY_1',
                    'SC_VELOCITY_2', 'SOLAR']

    Plotter(df = inputs_outputs_df, label = 'Inputs').df_plot_tiles(x_col = 'datetime', excluded_cols = col_range_raw + col_range, marker = ',', smoothing_key='smooth', show = False)
    Plotter(df = inputs_outputs_df, label = 'Outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = col_selected, marker = ',', smoothing_key='smooth', show = False)
    Plotter.show()
