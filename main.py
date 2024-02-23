"""Main file to run the project
Author: Andrea Adelfio
TODO:
- spostare funzioni di supporto nelle utils
"""
from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter
from sunmonitor import SunMonitor
import gc

########## BAT Catalog ##########
print('BAT Catalog...', end='')
cr = CatalogReader(start = 0, end = -1)
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
sco.open()
sc_params_df = sco.get_dataframe(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
sc_params_df = sco.filter_dataframe_with_run_times(sc_params_df, runs_times)
solar_signal_df = sco.filter_dataframe_with_run_times(solar_signal_df, runs_times)
inputs_df = sco.merge_dfs(sc_params_df, solar_signal_df)
inputs_outputs_df = sco.merge_dfs(tile_signal_df, inputs_df)

print(' done')
del sco, solar_signal_df
gc.collect
########### Plotting ############
print('Plotting...', end='')
# Y_KEY = 'LAT_GEO'
# Plotter(x = sc_params_df['START'], y = sc_params_df[Y_KEY], label = f'spacecraft LAT {Y_KEY}').plot(marker = ',', show = False)
Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
Plotter(df = inputs_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)
Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
Plotter.show()
print(' done')