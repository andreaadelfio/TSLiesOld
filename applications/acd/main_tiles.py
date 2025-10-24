#!/usr/bin/env python3
"""
main_tiles.py

Process trigger list CSV and extract per-tile 1s-binned, area-normalized time series for tiles belonging to triggered faces/bands.

Expect a triggers CSV with at least columns:
 - met_start (or start_met / start)
 - met_end   (or end_met / end)
 - face_band (or faces) : a string indicating face and band, examples:
     "top_low" or "Xpos_middle" or multiple separated by ";" e.g. "top_low;Xpos_high"
 - id (optional) : unique trigger id; if absent we will build one from row index

Workflow:
 - read `acd_runs.csv` (default from DATA_LATACD_PROCESSED_FOLDER_NAME/acd_runs.csv) to map a trigger MET to the run folder
 - for each trigger: identify run folder, build a TChain from .root files, select time window [met_start, met_end], build 1s bins, for each tile in the triggered faces compute counts per bin for the requested energy band, normalize by tile area
 - write a CSV per trigger in output folder with columns: MET, tile_<id> (one column per tile extracted)

Usage:
    python3 applications/acd/main_tiles.py triggers.csv [acd_runs.csv] [output_folder] [workers]

If optional arguments are omitted, defaults are used from `modules.config`.
"""
import os
import re
import sys
import math
import concurrent.futures
from functools import partial
from tqdm import tqdm

import pandas as pd
import numpy as np
import ROOT

from modules.config import DATA_LATACD_RAW_FOLDER_NAME, DATA_LATACD_PROCESSED_FOLDER_NAME
from modules.utils import Logger

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_SIZE_FILE = '/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/ACD_tiles_size2.txt'

logger = Logger('main_tiles').get_logger()

# mapping faces to tile indices (same as main_acd)
TILES_FACES = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}

# energy bands boundaries (in same spirit of main_acd)
ENERGY_BANDS = {
    'low': (0.0, 0.01),
    'middle': (0.01, 0.5),
    'high': (0.5, math.inf),
}


def resample_events(df, bin_edges, resample_rate):
    """
    Resample events to maintain statistical representation while reducing computation.
    Based on the same function in main_acd.py
    """
    resampled_indices = []
    for i in range(len(bin_edges) - 1):
        bin_events = df[(df['time'] >= bin_edges[i]) & (df['time'] < bin_edges[i + 1])]
        if len(bin_events) > 0:
            resample_prob = min(resample_rate / len(bin_events), 1.0)
            mask = np.random.rand(len(bin_events)) < resample_prob
            resampled_indices.extend(bin_events[mask].index)
    return df.loc[resampled_indices]


def load_tile_sizes(file_path=TILE_SIZE_FILE):
    d = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                tile = int(parts[0])
                area = float(parts[1])
                d[tile] = area
    except Exception as e:
        logger.error('Could not read tile size file %s: %s', file_path, e)
    return d


def find_run_for_met(acd_runs_df, met_start, met_end):
    """Find run row where met_value falls within run met_start-met_end (inclusive start, exclusive end).
    Returns the filename (folder name) or None.
    """
    # 1) exact containment: run fully contains trigger window
    cond = (acd_runs_df['met_start'] <= met_start) & (acd_runs_df['met_end'] >= met_end)
    rows = acd_runs_df[cond]
    if not rows.empty:
        return rows.iloc[0].to_dict()

    # 2) trigger start falls inside a run
    cond = (acd_runs_df['met_start'] <= met_start) & (acd_runs_df['met_end'] > met_start)
    rows = acd_runs_df[cond]
    if not rows.empty:
        return rows.iloc[0].to_dict()

    # 3) any overlap between run and trigger window -> pick the run with largest overlap
    cond = (acd_runs_df['met_start'] <= met_end) & (acd_runs_df['met_end'] >= met_start)
    rows = acd_runs_df[cond].copy()
    if not rows.empty:
        # overlap duration
        overlaps = np.minimum(rows['met_end'], met_end) - np.maximum(rows['met_start'], met_start)
        # keep only positive overlaps and choose the max
        overlaps[overlaps < 0] = 0
        idx = overlaps.idxmax()
        return rows.loc[idx].to_dict()

    return None


def parse_faces_column(face_col_value):
    """Parse face_band column. Accepts semicolon-separated list. Returns list of tuples (face, band).
    Examples: 'top_low' -> [('top','low')]
              'top_low;Xpos_high' -> [('top','low'),('Xpos','high')]
    """
    items = re.split(r'[;,\|]', str(face_col_value))
    out = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        # try face_band
        m = re.match(r'([A-Za-z0-9]+)_(low|middle|high)', it)
        if m:
            face = m.group(1)
            band = m.group(2)
            out.append((face, band))
        else:
            # allow just face -> default to all energies (None)
            out.append((it, None))
    return out


def build_chain_from_run_folder(run_folder):
    files = [f for f in os.listdir(run_folder) if f.endswith('.root')]
    files.sort()
    if not files:
        raise FileNotFoundError(f'No .root files in {run_folder}')
    chain = ROOT.TChain('myTree')
    for f in files:
        chain.Add(os.path.join(run_folder, f))
    return chain


def process_single_trigger(row, acd_runs_df, raw_base_folder, tile_sizes, output_folder):

    # faces column
    face_col = None
    if 'triggered_faces' in row:
        face_col = row['triggered_faces']
    if face_col is None:
        raise KeyError('No face/band column found in trigger row; expected face_band/faces')
    start_datetime = row['start_datetime']
    # id
    trig_id = f'{start_datetime[:-6]}_{face_col}'
    out_file = os.path.join(output_folder, f'{trig_id}.csv')
    if os.path.exists(out_file):
        logger.info('Trigger %s already processed, skipping', trig_id)
        return out_file
    """Process one trigger row (pandas Series or dict). Write CSV in output_folder. Return path or None on error."""
    # normalize keys
    if isinstance(row, pd.Series):
        row = row.to_dict()
    # ...existing code...
    # detect columns for met start/end (accept multiple common column names)
    for s_key in ('met_start', 'start_met', 'start'):
        if s_key in row:
            met_start = float(row[s_key])
            break
    else:
        raise KeyError('No met_start column found in trigger row')
    for e_key in ('met_end', 'stop_met', 'end_met', 'end'):
        if e_key in row:
            met_end = float(row[e_key])
            break
    else:
        raise KeyError('No met_end column found in trigger row')


    face_band_list = parse_faces_column(face_col)
    tiles_to_process = []
    for face, band in face_band_list:
        if face not in TILES_FACES:
            logger.warning('Unknown face "%s" in trigger %s, skipping', face, trig_id)
            continue
        tiles_range = TILES_FACES[face]
        # tiles are from tiles_range[0] to tiles_range[1]-1
        tiles = list(range(tiles_range[0], tiles_range[1]))
        tiles_to_process.append((tiles, band))

    if not tiles_to_process:
        logger.error('No valid tiles to process for trigger %s', trig_id)
        return None

    # find run
    run_row = find_run_for_met(acd_runs_df, met_start, met_end)
    if run_row is None:
        logger.error('No run found for trigger %s met %s', trig_id, met_start)
        return None
    run_folder_name = str(run_row.get('filename')).split('.')[0] or str(run_row.get('run')).split('.')[0]
    run_folder = os.path.join(raw_base_folder, '0' + run_folder_name)
    if not os.path.isdir(run_folder):
        logger.error('Run folder not found %s for trigger %s', run_folder, trig_id)
        return None

    # Load run data using RDataFrame -> AsNumpy (same approach as create_df)
    try:
        # build sorted list of root files
        list_files = [f for f in os.listdir(run_folder) if f.endswith('.root')]
        list_files.sort()
        if not list_files:
            logger.error('No root files in run folder %s for trigger %s', run_folder, trig_id)
            return None
        # create TChain and RDataFrame
        chain = ROOT.TChain('myTree')
        for f in list_files:
            chain.Add(os.path.join(run_folder, f))
        dict_np = ROOT.RDataFrame(chain).AsNumpy()
        df_root = pd.DataFrame(dict_np)
    except Exception as e:
        logger.error('Could not read RDataFrame for %s: %s', run_folder, e)
        return None

    # drop unused columns if present (mimic create_df)
    for col in ['gltGemSummary', 'gltGemEngine']:
        if col in df_root.columns:
            try:
                df_root.drop(columns=[col], inplace=True)
            except Exception:
                pass

    # filter rows to trigger time window
    df_window = df_root[(df_root['time'] >= met_start) & (df_root['time'] <= met_end)]
    if df_window.empty:
        logger.warning('No events in window for trigger %s in run %s', trig_id, run_folder_name)
        # still produce empty bins
    
    # prepare bins: 1s bins
    n_bins = int(max(1, math.ceil(met_end - met_start)))
    bin_edges = np.linspace(met_start, met_end, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Apply resampling like in create_df_norm to manage large datasets
    if not df_window.empty:
        binning = 1.0  # 1 second bins
        resample_rate = 200 * binning
        df_window = resample_events(df_window, bin_edges, resample_rate)
    
    result_df = pd.DataFrame({'MET': bin_centers})
    # for each tile list, compute per-tile histogram using numpy.histogram
    acd_arrs = df_window['acdE_acdtile'].values
    times = df_window['time'].values
    for tiles, band in tiles_to_process:
        lo, hi = (None, None)
        if band is not None:
            lo, hi = ENERGY_BANDS.get(band, (None, None))
        for tile in tiles:
            try:
                # acdE_acdtile is an array per event; extract energies for this tile
                if len(acd_arrs) == 0:
                    print('No events found')
                    counts = np.zeros(n_bins, dtype=float)
                else:
                    acdE_acdtile = np.array([evt[tile] for evt in acd_arrs])
                    if band is None:
                        mask = acdE_acdtile > 0
                    else:
                        if hi == math.inf:
                            mask = acdE_acdtile > lo
                        else:
                            mask = (acdE_acdtile > lo) & (acdE_acdtile <= hi)
                    times_sel = times[mask]
                    if len(times_sel) == 0:
                        counts = np.zeros(n_bins, dtype=float)
                    else:
                        counts, _ = np.histogram(times_sel, bins=bin_edges)
                area = tile_sizes.get(tile, 1.0)
                vals = counts.astype(float) / 1 
                result_df[f'tile_{tile}'] = vals
            except Exception as e:
                logger.error('Error processing tile %s for trigger %s: %s', tile, trig_id, e)
                result_df[f'tile_{tile}'] = np.zeros(n_bins, dtype=float)
    # ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    result_df.to_csv(out_file, index=False)
    logger.info('Wrote trigger %s -> %s', trig_id, out_file)
    return out_file


def _worker_process_trigger(args):
    # thin wrapper for ProcessPoolExecutor
    return process_single_trigger(*args)


def process_triggers_file(triggers_csv, acd_runs_csv=None, raw_base_folder=None, output_folder=None, workers=4):
    acd_runs_csv = acd_runs_csv or os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'acd_runs.csv')
    raw_base_folder = DATA_LATACD_RAW_FOLDER_NAME

    if not os.path.exists(triggers_csv):
        raise FileNotFoundError(f'Triggers file not found: {triggers_csv}')
    if not os.path.exists(acd_runs_csv):
        raise FileNotFoundError(f'acd_runs.csv not found: {acd_runs_csv}')

    triggers_df = pd.read_csv(triggers_csv)
    acd_runs_df = pd.read_csv(acd_runs_csv)

    tile_sizes = load_tile_sizes()

    tasks = []
    for idx, row in triggers_df.iterrows():
        tasks.append((row, acd_runs_df, raw_base_folder, tile_sizes, output_folder))

    results = []
    if workers and workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker_process_trigger, t) for t in tasks]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing triggers'):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logger.error('Worker failed: %s', e)
    else:
        print('Processing triggers sequentially...')
        for t in tqdm(tasks, desc='Processing triggers'):
            try:
                res = process_single_trigger(*t)
                results.append(res)
            except Exception as e:
                logger.error('Processing trigger failed: %s', e)

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 main_tiles.py triggers.csv [acd_runs.csv] [output_folder] [workers]')
    triggers_csv = 'best/anomalies/MA 7/plots/total_detections.csv'
    acd_runs_csv = 'data/LAT_ACD/processed/new_with_correct_triggs/acd_runs.csv'
    output_folder = 'best/anomalies/MA 7/tiles_triggers'
    workers = 4
    res = process_triggers_file(triggers_csv, acd_runs_csv=acd_runs_csv, output_folder=output_folder, workers=workers)
    print('Done. Generated files:')
    for r in res:
        print(r)
