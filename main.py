from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter

########## BAT Catalog ##########
cr = CatalogReader(from_lat = True, start = 0, end = 2)
runs_roots = cr.get_runs_roots()
runs_dict = cr.get_runs_dict(runs_roots, smooth = False)
runs_times = cr.get_runs_times()
bkg_signal_df = cr.get_signal_df_from_catalog(runs_dict)
# for key, signals in runs_dict.items():
#     label = f'{key}, met {runs_times[key][0]}'
#     Plotter(xy = signals, label = label).plot_tiles(lw = 0.5, with_smooth = True)

############## GBM ##############
# sco = SpacecraftOpener()
# sco.get_from_gbm(runs_times.values())
# sco.get_from_lat_weekly()
# bkg_signal_df = sco.get_from_lat_weekly_poshist(bkg_signal_df)
# sco.open(from_gbm = True)
# initial_data = sco.get_data()
# sc_input_df = sco.get_sc_input_dataframe(initial_data, runs_times)

############## LAT ##############
sco = SpacecraftOpener()
sco.open()
initial_data = sco.get_data()
sc_input_df = sco.get_sc_input_dataframe(initial_data, runs_times)
Y_KEY = 'LAT_GEO'
Plotter(x = sc_input_df['START'], y = sc_input_df[Y_KEY], label = f'spacecraft LAT {Y_KEY}').plot(marker = ',', show = False)
Plotter(df = bkg_signal_df, label = 'All signals').df_plot_tiles(x_column = 'time', marker = ',', show = False)
Plotter.show()