"""Main file to run the project
Author: Andrea Adelfio
TODO:
- aggiungere funzione per scaricare sc data con wget
"""
from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter
from sunmonitor import SunMonitor
from utils import Time, Data
import gc

def fetch_and_write_inputs_outputs():
    ########## BAT Catalog ##########
    print('BAT Catalog...', end='')
    cr = CatalogReader(start = 0, end = 2)
    tile_signal_df = cr.get_signal_df_from_catalog()
    tile_signal_df = cr.add_smoothing(tile_signal_df)
    runs_times = cr.get_runs_times()

    print(' done')
    del cr
    gc.collect()
    ############## GOES ##############
    print('GOES data...', end='')
    tstart, tend = list(runs_times.values())[0][0], list(runs_times.values())[-1][1]
    sm = SunMonitor(tstart, tend)
    file_goes = sm.fetch_goes_data()
    solar_signal_df = sm.find_goes_data(file_goes)

    print(' done')
    del sm, file_goes
    gc.collect
    ############## LAT ##############
    print('LAT data...', end='')
    sco = SpacecraftOpener()
    sco.open(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
    sc_params_df = sco.get_dataframe()
    sc_params_df = Data.filter_dataframe_with_run_times(sc_params_df, runs_times)
    solar_signal_df = Data.filter_dataframe_with_run_times(solar_signal_df, runs_times)
    inputs_df = Data.merge_dfs(sc_params_df, solar_signal_df)
    inputs_outputs_df = Data.merge_dfs(tile_signal_df, inputs_df)
    Data.write_df_on_file(inputs_outputs_df)

    print(' done')
    del sco, solar_signal_df
    gc.collect
    return tile_signal_df, inputs_df, inputs_outputs_df


########### Main ############
if __name__ == '__main__':
    inputs_outputs_df = Data.read_df_from_file()
    if inputs_outputs_df is None:
        tile_signal_df, inputs_df, inputs_outputs_df = fetch_and_write_inputs_outputs()
        # Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
        # Plotter(df = inputs_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)

    Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
    ########### Plotting ############
    print('Plotting...', end='')
    # Y_KEY = 'LAT_GEO'
    # Plotter(x = sc_params_df['START'], y = sc_params_df[Y_KEY], label = f'spacecraft LAT {Y_KEY}').plot(marker = ',', show = False)
    Plotter.show()

    print(' done')