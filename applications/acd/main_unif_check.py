#!/usr/bin/env python3
"""
main_unif_check.py

Analizza i CSV generati da main_tiles per valutare l'uniformità delle hit nelle tiles
delle facce triggerate.

Score di uniformità basato su coefficiente di variazione inverso:
- [0, 0, 0, 8, 0] -> non uniforme, score basso
- [3, 5, 7, 4, 5] -> uniforme, score alto  
- [3, 2, 3, 9, 2] -> non uniforme, score basso

Usage:
    python3 applications/acd/main_unif_check.py
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

from modules.config import DATA_LATACD_PROCESSED_FOLDER_NAME
from modules.utils import Logger

logger = Logger('main_unif_check').get_logger()

TILES_FACES = {'top': (64, 89), 'Xpos': (48, 64), 'Xneg': (32, 48), 'Ypos': (16, 32), 'Yneg': (0, 16)}


def calculate_uniformity_score(values, method='cv_inverse'):
    """
    Calcola score di uniformità per un array di valori.
    
    Args:
        values: array di valori numerici
        method: metodo di calcolo ('cv_inverse', 'entropy', 'gini_inverse')
    
    Returns:
        float: score di uniformità (più alto = più uniforme)
    """
    values = np.array(values, dtype=float)
    
    values = values[values >= 0]
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return 0.0
    
    if method == 'cv_inverse':
        if np.sum(values) == 0:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        score = 1.0 / (1.0 + cv)
        return score


def extract_face_tiles_from_columns(df_columns):
    """
    Estrae mapping face -> tile_columns dal DataFrame.
    
    Args:
        df_columns: colonne del DataFrame (es. ['MET', 'tile_64', 'tile_65', ...])
    
    Returns:
        dict: {face: [tile_columns]}
    """
    face_tiles = {}
    
    for face, (start_idx, end_idx) in TILES_FACES.items():
        tiles_cols = []
        for tile_idx in range(start_idx, end_idx):
            col_name = f'tile_{tile_idx}'
            if col_name in df_columns:
                tiles_cols.append(col_name)
        
        if tiles_cols:
            face_tiles[face] = tiles_cols
    
    return face_tiles


def analyze_trigger_csv(csv_path, uniformity_method='cv_inverse'):
    """
    Analizza un singolo CSV di trigger per calcolare score di uniformità.
    
    Args:
        csv_path: percorso al file CSV
        uniformity_method: metodo per calcolo uniformità
    
    Returns:
        dict: risultati analisi
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Errore lettura CSV {csv_path}: {e}")
        return None
    
    if df.empty:
        logger.warning(f"CSV vuoto: {csv_path}")
        return None
    
    face_tiles = extract_face_tiles_from_columns(df.columns)
    
    if not face_tiles:
        logger.warning(f"Nessuna tile trovata in {csv_path}")
        return None
    
    results = []
    trigger_id = os.path.splitext(os.path.basename(csv_path))[0]
    
    for idx, row in df.iterrows():
        met_time = row.get('MET', idx)
        
        for face, tile_columns in face_tiles.items():
            tile_values = [row[col] for col in tile_columns]
            
            total_hits = np.sum(tile_values)
            
            if total_hits == 0:
                continue
            
            uniformity_score = calculate_uniformity_score(tile_values, method=uniformity_method)
            
            max_hits = np.max(tile_values)
            n_active_tiles = np.sum(np.array(tile_values) > 0)
            
            results.append({
                'trigger_id': trigger_id,
                'met': met_time,
                'face': face,
                'uniformity_score': uniformity_score,
                'total_hits': total_hits,
                'max_hits': max_hits,
                'n_active_tiles': n_active_tiles,
                'n_total_tiles': len(tile_columns)
            })
    return results


def analyze_tiles_folder(tiles_folder, uniformity_method='cv_inverse', output_csv=None):
    """
    Analizza tutti i CSV nella cartella tiles per uniformità.
    
    Args:
        tiles_folder: cartella contenente i CSV dei trigger
        uniformity_method: metodo per calcolo uniformità
        output_csv: percorso file CSV output (opzionale)
    
    Returns:
        pd.DataFrame: risultati aggregati
    """
    csv_files = glob.glob(os.path.join(tiles_folder, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV trovato in {tiles_folder}")
    
    logger.info(f"Trovati {len(csv_files)} file CSV da analizzare")
    
    all_results = []
    
    for csv_path in tqdm(csv_files, desc="Analizzando trigger"):
        results = analyze_trigger_csv(csv_path, uniformity_method)
        if results:
            all_results.extend(results)
    
    if not all_results:
        raise ValueError("Nessun risultato valido ottenuto dall'analisi")
    
    df_results = pd.DataFrame(all_results)
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        logger.info(f"Risultati salvati in {output_csv}")
    return df_results


def generate_uniformity_histogram(df_results, output_plot=None, bins=30):
    """
    Genera istogramma degli score di uniformità e total hits.
    
    Args:
        df_results: DataFrame con risultati analisi
        output_plot: percorso file plot (opzionale)
        bins: numero di bin per istogramma
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Uniformity vs Total Activity
    plt.subplot(2, 3, 1)
    plt.scatter(df_results['total_hits'], df_results['uniformity_score'], alpha=0.5)
    plt.xlabel('Total Hits')
    plt.ylabel('Uniformity Score')
    plt.title('Uniformity vs Total Activity')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of Uniformity Scores
    plt.subplot(2, 3, 2)
    counts, bin_edges = np.histogram(df_results['uniformity_score'], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.barh(bin_centers, counts, height=bin_edges[1] - bin_edges[0], alpha=0.7, edgecolor='black')
    plt.ylabel('Uniformity Score')
    plt.xlabel('Frequency')
    plt.title('Distribution of Uniformity Scores')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Uniformity Score by Face
    plt.subplot(2, 3, 3)
    faces = df_results['face'].unique()
    for face in faces:
        face_data = df_results[df_results['face'] == face]['uniformity_score']
        plt.hist(face_data, bins=bins//2, alpha=0.6, label=face, edgecolor='black', orientation='horizontal', linewidth=0.8)
    plt.xlabel('Uniformity Score')
    plt.ylabel('Frequency')
    plt.title('Uniformity Score by Face')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of Total Hits
    plt.subplot(2, 3, 4)
    plt.hist(df_results['total_hits'], bins=bins, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Total Hits')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Hits')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Total Hits per Face
    plt.subplot(2, 3, 5)
    for face in faces:
        face_data = df_results[df_results['face'] == face]['total_hits']
        plt.hist(face_data, bins=bins//2, alpha=0.6, label=face, edgecolor='black', linewidth=0.8)
    plt.xlabel('Total Hits')
    plt.ylabel('Frequency')
    plt.title('Total Hits by Face')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_plot:
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        logger.info(f"Plot salvato in {output_plot}")
    else:
        plt.show()
    
    plt.close()


def apply_cut(df_results, tiles_folder, threshold=0.25, filter_name=None, operator='<', output_subfolder='low_uniformity_triggers', total_detections=None, total_detections_clean=None, comparison_results_csv=None, comparison_results_csv_clean=None):
    """
    Filtra il DataFrame e copia i CSV dei trigger che soddisfano la condizione.
    
    Args:
        df_results: DataFrame con risultati analisi
        tiles_folder: cartella contenente i CSV originali
        threshold: soglia di confronto
        filter_name: nome della colonna su cui applicare il filtro
        operator: operatore di confronto ('<', '>', '<=', '>=', '==', '!=')
        output_subfolder: nome sottocartella di destinazione
    
    Returns:
        pd.DataFrame: DataFrame filtrato
    """
    # Applica il filtro in base all'operatore
    if operator == '<':
        filtered_df = df_results[df_results[filter_name] < threshold]
    elif operator == '>':
        filtered_df = df_results[df_results[filter_name] > threshold]
    elif operator == '<=':
        filtered_df = df_results[df_results[filter_name] <= threshold]
    elif operator == '>=':
        filtered_df = df_results[df_results[filter_name] >= threshold]
    elif operator == '==':
        filtered_df = df_results[df_results[filter_name] == threshold]
    elif operator == '!=':
        filtered_df = df_results[df_results[filter_name] != threshold]
    else:
        raise ValueError(f"Operatore non supportato: {operator}")

    filtered_triggers = filtered_df['trigger_id'].unique()
    
    if len(filtered_triggers) == 0:
        logger.info(f"Nessun trigger con {filter_name} {operator} {threshold}")
        return df_results  # Restituisce il DataFrame originale se nessun trigger soddisfa il filtro
    
    logger.info(f"Trovati {len(filtered_triggers)} trigger con {filter_name} {operator} {threshold}")
    
    output_folder = os.path.join(tiles_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    copied_files = []
    
    for trigger_id in tqdm(filtered_triggers, desc=f"Copiando trigger con {filter_name} {operator} {threshold}"):
        csv_pattern = os.path.join(tiles_folder, f"{trigger_id}.csv")
        matching_files = glob.glob(csv_pattern)
        
        if not matching_files:
            csv_pattern = os.path.join(tiles_folder, f"*{trigger_id}*.csv")
            matching_files = glob.glob(csv_pattern)
        
        for src_file in matching_files:
            dst_file = os.path.join(output_folder, os.path.basename(src_file))
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append(dst_file)
                logger.debug(f"Copiato: {src_file} -> {dst_file}")
            except Exception as e:
                logger.warning(f"Errore copia {src_file}: {e}")
    
    if total_detections  and comparison_results_csv:
        total_detections_clean = os.path.join(os.path.dirname(os.path.dirname(total_detections)), total_detections_clean)
        comparison_results_csv_clean = os.path.join(os.path.dirname(os.path.dirname(comparison_results_csv)), comparison_results_csv_clean)
        try:
            df_original = pd.read_csv(total_detections)
            df_comparison = pd.read_csv(comparison_results_csv)
            for idx, row in df_original.iterrows():
                df_original.at[idx, 'trigger_id'] = str(row['start_datetime']).split('+')[0] + '_' + row['triggered_faces']
            for idx, row in df_comparison.iterrows():
                df_comparison.at[idx, 'trigger_id'] = str(row['det_start_datetime']).split('+')[0] + '_' + row['det_faces']
            df_filtered_final = df_original[df_original['trigger_id'].isin(filtered_df['trigger_id'])]
            df_filtered_final_comparison = df_comparison[df_comparison['trigger_id'].isin(filtered_df['trigger_id'])]
            df_filtered_final.to_csv(total_detections_clean, index=False)
            df_filtered_final_comparison.to_csv(comparison_results_csv_clean, index=False)
            logger.info(f"File finale filtrato salvato in {total_detections_clean}")
        except Exception as e:
            logger.error(f"Errore durante il filtraggio del file finale: {e}")

    return filtered_df


def main():
    """Main function per analisi uniformità tiles."""
    tiles_folder = 'best/anomalies/MA 7/tiles_triggers'
    output_csv = 'best/anomalies/MA 7/uniformity_analysis.csv'
    output_plot = 'best/anomalies/MA 7/uniformity_histogram.png'
    final_output_plot = 'best/anomalies/MA 7/uniformity_histogram_final.png'
    total_detections = 'best/anomalies/MA 7/plots/total_detections.csv'
    total_detections_clean = 'total_detections_clean.csv'
    comparison_results_csv = 'best/anomalies/MA 7/plots/comparison_results.csv'
    comparison_results_csv_clean = 'comparison_results_clean.csv'
    try:
        logger.info(f"Avvio analisi uniformità su cartella: {tiles_folder}")
        df_results = analyze_tiles_folder(tiles_folder, output_csv=output_csv)
        
        logger.info("Generazione istogramma uniformità...")
        generate_uniformity_histogram(df_results, output_plot=output_plot)
        
        logger.info("Applicazione filtro...")
        df_results = apply_cut(df_results, tiles_folder, threshold=40, filter_name='total_hits', operator='>', total_detections=total_detections, total_detections_clean=total_detections_clean, comparison_results_csv=comparison_results_csv, comparison_results_csv_clean=comparison_results_csv_clean)
        
        generate_uniformity_histogram(df_results, output_plot=final_output_plot)
        logger.info("Analisi completata con successo!")

    except Exception as e:
        logger.error(f"Errore durante analisi: {e}")
        print(f"Errore: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()