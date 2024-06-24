'''Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 17/05/2024
TODO:
'''
from modules.spacecraftopener import SpacecraftOpener
from modules.catalogreader import CatalogReader
from modules.plotter import Plotter
from modules.sunmonitor import SunMonitor
from modules.utils import Data, File, Time, Logger, logger_decorator
from modules.config import INPUTS_OUTPUTS_FILE_PATH


logger = Logger('Main Dataset').get_logger()


@logger_decorator(logger)
def get_tiles_signal_df():
    '''Get the tile signal dataframe from the catalog'''
    print('Catalog...', end='')
    cr = CatalogReader(data_dir='data/LAT_ACD/output runs v2', start=0, end=-1)
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
    Plotter(df = tile_signal_df, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
                                                                    excluded_cols = [],
                                                                    marker = ',',
                                                                    smoothing_key='smooth',
                                                                    show = True)
    saa_exit_time = 0
    for week in [week for week in weeks_list if week not in ()]:
        sc_params_df, saa_exit_time = get_sc_params_df(week, saa_exit_time)
        inputs_outputs = Data.merge_dfs(tile_signal_df, sc_params_df)
        solar_signal_df = get_solar_signal_df(week)
        inputs_outputs = Data.merge_dfs(inputs_outputs, solar_signal_df)
        # Plotter(df = inputs_outputs, label = 'Inputs').df_plot_tiles(x_col = 'datetime',
        #                                                                 excluded_cols = [],
        #                                                                 marker = ',',
        #                                                                 smoothing_key='smooth',
        #                                                                 show = True)
        File.write_df_on_file(inputs_outputs,
                              filename=f'{INPUTS_OUTPUTS_FILE_PATH}_w{week}',
                              fmt='both')
    print(' done')
    return inputs_outputs


# MARK: Main
if __name__ == '__main__':
    inputs_outputs_df = get_inputs_outputs_df()
    # inputs_outputs_df = File.read_dfs_from_pk_folder()
    # # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    # #                                               start='2023-12-07 04:00:00',
    # #                                               stop='2023-12-08 04:00:00')
    # # File.write_df_on_file(inputs_outputs_df, './inputs_outputs_df')


    # col_range_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    # col_range = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    # # col_selected = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
    # #                 'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 
    # #                 'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ', 'START', 'STOP', 'MET', 'IN_SAA',
    # #                 'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'LIVETIME', 'DEC_SCZ', 'RA_SCX',
    # #                 'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE', 'QSJ_1', 'QSJ_2',
    # #                 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN', 'SC_VELOCITY_0', 'SC_VELOCITY_1',
    # #                 'SC_VELOCITY_2', 'SOLAR']
    # col_selected = inputs_outputs_df.columns

    Plotter(df = inputs_outputs_df, label = 'Outputs').df_plot_tiles(x_col = 'datetime',
                                                                     excluded_cols = [],
                                                                     marker = ',', smoothing_key='smooth',
                                                                     show = False)
    Plotter.show()
