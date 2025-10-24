#!/usr/bin/env python3
"""
main_checks.py
Utility to scan ACD raw run folders and produce a CSV with: folder name, run id, MET start, MET end

Usage:
    python main_checks.py [raw_folder] [output_csv]

If arguments are omitted, uses values from modules.config:
    DATA_LATACD_RAW_FOLDER_NAME and DATA_LATACD_PROCESSED_FOLDER_NAME
"""
import os
import re
import sys
import pandas as pd
import ROOT
import concurrent.futures
from tqdm import tqdm

from modules.config import DATA_LATACD_RAW_FOLDER_NAME, DATA_LATACD_PROCESSED_FOLDER_NAME
from modules.utils import Logger

logger = Logger('main_checks').get_logger()


def get_run_met_range(run_path, tree_name='myTree'):
    """Return (met_start, met_end) for the chain built from .root files in run_path.

    Raises ValueError if no root files or tree has no entries.
    """
    files = [f for f in os.listdir(run_path) if f.endswith('.root')]
    if not files:
        raise ValueError(f'No .root files in {run_path}')

    # sort by first number in filename if present, otherwise lexicographic
    def _sort_key(fname):
        m = re.search(r"\d+", fname)
        return int(m.group(0)) if m else fname

    files.sort(key=_sort_key)

    # build chain and use RDataFrame -> AsNumpy -> pandas DataFrame to get min/max time
    chain = ROOT.TChain(tree_name)  # pylint: disable=maybe-no-member
    for f in files:
        chain.Add(os.path.join(run_path, f))

    try:
        dict_np = ROOT.RDataFrame(chain).AsNumpy()
        df = pd.DataFrame(dict_np)
    except Exception as e:
        raise ValueError(f'Could not build dataframe from ROOT files in {run_path}: {e}')

    if df.empty:
        raise ValueError(f'No entries in tree for {run_path}')

    # drop columns that are not needed (mirrors create_df behavior)
    for col in ('gltGemSummary', 'gltGemEngine'):
        if col in df.columns:
            try:
                df.drop(columns=[col], inplace=True)
            except Exception:
                pass

    if 'time' not in df.columns:
        raise ValueError(f"'time' branch not found in tree for {run_path}")

    time0 = float(df['time'].min())
    time_last = float(df['time'].max())
    return time0, time_last


def _process_run_entry(args):
    """Helper top-level function for parallel processing (picklable).
    args: tuple(entry, raw_folder)
    returns dict or None on error
    """
    entry, raw_folder = args
    run_path = os.path.join(raw_folder, entry)
    m = re.search(r"\d+", entry)
    run = m.group(0) if m else entry
    try:
        t0, t1 = get_run_met_range(run_path)
        return {'filename': entry, 'run': run, 'met_start': t0, 'met_end': t1}
    except Exception as e:
        # return error info to be handled by caller
        return {'filename': entry, 'run': run, 'error': str(e)}


def extract_runs_csv(raw_folder=None, output_csv=None, workers=4):
    """Scan run folders under raw_folder and write CSV with columns:
    filename, run, met_start, met_end
    If workers>1 uses ProcessPoolExecutor and shows a tqdm progress bar.
    Returns path to CSV file written.
    """
    raw_folder = raw_folder or DATA_LATACD_RAW_FOLDER_NAME
    output_csv = output_csv or os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'acd_runs.csv')

    if not os.path.exists(raw_folder):
        raise FileNotFoundError(f'Raw folder not found: {raw_folder}')

    entries = []
    for entry in sorted(os.listdir(raw_folder)):
        run_path = os.path.join(raw_folder, entry)
        if not os.path.isdir(run_path):
            continue
        root_files = [f for f in os.listdir(run_path) if f.endswith('.root')]
        if not root_files:
            continue
        entries.append(entry)

    rows = []
    # parallel processing
    if workers and workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_run_entry, (entry, raw_folder)): entry for entry in entries}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Scanning runs'):
                res = fut.result()
                if res is None:
                    continue
                if 'error' in res:
                    logger.error('Error processing %s: %s', res.get('filename'), res.get('error'))
                    try:
                        with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a', encoding='utf-8') as ef:
                            ef.write(f"Error {res.get('error')} in {os.path.join(raw_folder, res.get('filename'))}\n")
                    except Exception:
                        pass
                else:
                    rows.append(res)
    else:
        for entry in tqdm(entries, desc='Scanning runs'):
            run_path = os.path.join(raw_folder, entry)
            m = re.search(r"\d+", entry)
            run = m.group(0) if m else entry
            try:
                t0, t1 = get_run_met_range(run_path)
                rows.append({'filename': entry, 'run': run, 'met_start': t0, 'met_end': t1})
            except Exception as e:
                logger.error('Error processing %s: %s', entry, e)
                try:
                    with open(os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, 'countrates_errors.txt'), 'a', encoding='utf-8') as ef:
                        ef.write(f'Error {e} in {run_path}\n')
                except Exception:
                    pass

    df = pd.DataFrame(rows, columns=['filename', 'run', 'met_start', 'met_end'])
    # ensure output dir exists
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    logger.info('Wrote %d runs to %s', len(df), output_csv)
    return output_csv


if __name__ == '__main__':
    raw = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        csv_path = extract_runs_csv(raw, out)
        print(f'Run summary written to: {csv_path}')
    except Exception as e:
        logger.error(f'Extraction failed: {e}')
        print(f'Extraction failed: {e}', file=sys.stderr)
        sys.exit(1)
