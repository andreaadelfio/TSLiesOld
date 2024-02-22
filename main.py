from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter
from sunmonitor import find_goes_data
from utils import from_met_to_datetime_str, from_met_to_datetime

########## BAT Catalog ##########
cr = CatalogReader(from_lat = True, start = 0, end = -1)
runs_roots = cr.get_runs_roots()
runs_dict = cr.get_runs_dict(runs_roots)
runs_times = cr.get_runs_times()
tile_signal_df = cr.get_signal_df_from_catalog(runs_dict)
tile_signal_df = cr.add_smoothing(tile_signal_df)

############## GOES ##############
tstart, tend = list(runs_times.values())[0][0], list(runs_times.values())[-1][1]
solar_signal_df = find_goes_data(tstart=tstart, tend=tend)

############## LAT ##############
sco = SpacecraftOpener()
sco.open()
initial_data = sco.get_data(excluded_columns = ['LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'IN_SAA', 'STOP', 'LIVETIME'])
sc_params_df = sco.convert_to_df(initial_data)
sc_params_df = sco.filter_dataframe_with_run_times(sc_params_df, runs_times)
solar_signal_df = sco.filter_dataframe_with_run_times(solar_signal_df, runs_times)
inputs_df = sco.merge_dfs(sc_params_df, solar_signal_df)
inputs_outputs_df = sco.merge_dfs(tile_signal_df, inputs_df)

########### Plotting ############
# Y_KEY = 'LAT_GEO'
# Plotter(x = sc_params_df['START'], y = sc_params_df[Y_KEY], label = f'spacecraft LAT {Y_KEY}').plot(marker = ',', show = False)
Plotter(df = tile_signal_df, label = 'Tiles signals').df_plot_tiles(x_col = 'datetime', excluded_cols = ['MET'], marker = ',', show = False, with_smooth = True)
Plotter(df = inputs_df, label = 'Inputs (SC + solar activity)').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START'], marker = ',', show = False)
Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = ['START', 'MET'], marker = ',', show = False, with_smooth = True)
Plotter.show()