import pandas as pd
from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter

########## BAT Catalog ##########
cr = CatalogReader(start = 0, end = 2)
events_dirs = cr.get_events_dirs()
event_dict = cr.get_event_dict(events_dirs, smooth = False)
event_times = cr.get_event_times()
print(event_times)
bkg_signal_df = cr.get_signal_df_from_catalog(event_dict)
# for key, signals in event_dict.items():
#     label = f'{key}, met {event_times[key][0]}'
#     Plotter(xy = signals, label = label).plot_tiles(lw = 0.5, with_smooth = True)

############## GBM ##############
sco = SpacecraftOpener()
sco.get_from_gbm(event_times.values())
bkg_signal_df = sco.get_from_gbm_poshist(bkg_signal_df)
# sco.open(from_gbm = True)
# initial_data = sco.get_data()
# sc_input_df = sco.get_sc_input_dataframe(initial_data, event_times)


############## LAT ##############
sco = SpacecraftOpener()
sco.open()
initial_data = sco.get_data()
sc_input_df = sco.get_sc_input_dataframe(initial_data, event_times)
Y_KEY = 'LAT_GEO'
Plotter(x = sc_input_df['START'], y = sc_input_df[Y_KEY], label = f'spacecraft LAT {Y_KEY}').plot(marker = ',', show = False)
Plotter(df = bkg_signal_df, label = 'All signals').df_plot_tiles(x_column = 'time', marker = ',', show = False)
Plotter.show()