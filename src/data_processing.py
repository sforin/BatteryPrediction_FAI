import os
import glob
import pandas as pd
import numpy as np

import os
import glob
import pandas as pd
import numpy as np

def transform_single_csv(file_path, lags=4, rolling_window=5, resample_sec=None):
    """
    Legge un singolo file CSV, calcola elapsed_min ed espande le feature.
    Sottocampiona i dati se resample_sec è impostato per alleggerire il carico.
    """
    df = pd.read_csv(file_path, sep=';')
    
    df['Tempo'] = pd.to_datetime(df['Tempo'], utc=True)
    df = df.sort_values('Tempo').reset_index(drop=True)
    
    # Calcolo immediato del tempo trascorso in minuti dall'inizio della scarica
    df['elapsed_min'] = (df['Tempo'] - df['Tempo'].iloc[0]).dt.total_seconds() / 60.0

    # Opzionale: Downsampling a monte per ridurre le righe (es. aggregazione ogni 30s)
    if resample_sec is not None and resample_sec > 1:
        df = df.set_index('Tempo').resample(f'{resample_sec}s').mean().reset_index()
        # Ricalcola elapsed_min dopo il resample se necessario
        df['elapsed_min'] = (df['Tempo'] - df['Tempo'].iloc[0]).dt.total_seconds() / 60.0

    if len(df) <= (lags + rolling_window):
        return None

    # Feature Engineering standard (Volt, Corrente, Capacità)
    df['V_mean'] = df['Tensione (V)'].rolling(window=rolling_window).mean()
    df['V_std'] = df['Tensione (V)'].rolling(window=rolling_window).std()
    df['dV'] = df['Tensione (V)'].diff()
    
    # Riempi i primi valori NaN generati da rolling/diff
    df = df.bfill().ffill()
    return df

def extract_features_at_cutoff(df_single_cell, max_minutes, lags=4):
    """
    Prende i dati di UNA cella filtrati fino al minuto max_minutes e restituisce 
    UNA SOLA RIGA riassuntiva di feature (V_mean, dV, ecc.) per quella finestra.
    Evita la ridondanza di avere migliaia di righe per una sola cella.
    """
    # Filtro cumulativo: prendiamo solo i dati dall'inizio fino al limite stabilito W
    df_w = df_single_cell[df_single_cell['elapsed_min'] <= max_minutes].copy()
    
    if len(df_w) == 0:
        return None
        
    # Estraiamo indicatori statistici globali di questa prima porzione di scarica
    features = {
        'elapsed_min_cutoff': max_minutes,
        'V_last': df_w['Tensione (V)'].iloc[-1],
        'V_start': df_w['Tensione (V)'].iloc[0],
        'V_mean_window': df_w['Tensione (V)'].mean(),
        'V_std_window': df_w['Tensione (V)'].std(),
        'dV_total': df_w['Tensione (V)'].iloc[-1] - df_w['Tensione (V)'].iloc[0],
        'C_accumulated': df_w['Capacity (Ah)'].iloc[-1]
    }
    
    # Aggiungiamo dinamicamente gli ultimi 'lags' punti registrati fino a quel momento
    for l in range(1, lags + 1):
        if len(df_w) > l:
            features[f'V_lag_{l}'] = df_w['Tensione (V)'].iloc[-1 - l]
        else:
            features[f'V_lag_{l}'] = df_w['Tensione (V)'].iloc[0]
            
    return pd.DataFrame([features])

def build_global_dataset(raw_data_dir, output_dir, lags=4, rolling_window=5):
    """
    Iterate through all the CSV files in the raw data folder, apply the 
    pure transformation to each one, concatenate the results into a single large dataset,
    and save it to the processed data folder in Parquet format.
    """
    search_path = os.path.join(raw_data_dir, "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_data_dir}")
        
    print(f"Found {len(csv_files)} test files to process...")
    
    all_processed_dfs = []
    
    for file_path in csv_files:
        try:
            processed_df = transform_single_csv(
                file_path, 
                lags=lags, 
                rolling_window=rolling_window
            )
            if processed_df is not None:
                all_processed_dfs.append(processed_df)
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            
    if not all_processed_dfs:
        raise ValueError("No data was successfully processed. Check the CSV structure.")
        
    # Concatenation of all transformed tests into a single large structured matrix
    global_dataset = pd.concat(all_processed_dfs, ignore_index=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in Parquet format to preserve data types and optimize space
    output_file_path = os.path.join(output_dir, "dataset_features.parquet")
    global_dataset.to_parquet(output_file_path, index=False)
    
    print(f"Processing completed successfully!")
    print(f"Expanded global dataset dimensions: {global_dataset.shape}")
    print(f"Dataset saved in: {output_file_path}")
    
    return global_dataset

def create_time_windows(expanded_df, window_size_steps=600, step_size_steps=300):
    """
    It takes the pre-expanded feature matrix of a single file (the output of `transform_single_csv`)
    and applies a sliding window for data augmentation.
    
    It returns a DataFrame containing all the concatenated sub-windows for that file,
    adding a 'window_id' column to uniquely identify each fragment.
    """
    windows = []
    total_steps = len(expanded_df)
    window_counter = 0
    start_idx = 0
    
    # Scroll through the time series until there is space for a complete window
    while start_idx + window_size_steps <= total_steps:
        # Extract the block of rows (the window)
        window_df = expanded_df.iloc[start_idx : start_idx + window_size_steps].copy()
        
        # Assign a unique identifier that links the window to the source file
        file_name = window_df['file_name'].iloc[0]
        window_df['window_id'] = f"{file_name}_w{window_counter}"
        
        windows.append(window_df)
        
        # Advance by the specified step (sliding step)
        start_idx += step_size_steps
        window_counter += 1
        
    if not windows:
        return None
        
    return pd.concat(windows, ignore_index=True)

def build_augmented_dataset(
        raw_data_dir,
        output_dir,
        lags=4,
        rolling_window=5,
        window_size_min=10,
        step_size_min=5,
        sampling_rate_sec=1,
        resample_sec=None):
    """
    Orchestrator of Data Augmentation:
    1. Loops through all raw CSV files in the 'raw_data_dir' folder.
    2. Expands features row by row (Lag, Derivatives, Power) using 'transform_single_csv'.
    3. Splits the resulting curve into temporal sub-windows using 'create_time_windows'.
    4. Concatenates everything into a single global matrix and saves it in Parquet format.

    Parameters:
    - resample_sec: If set, aggregate raw telemetry before feature engineering.
                    Set sampling_rate_sec to the same value when downsampling
                    so window_size_steps is computed correctly.
    """
    # Convert the requested minutes into the number of rows/steps based on the sampling rate
    window_size_steps = int((window_size_min * 60) / sampling_rate_sec)
    step_size_steps = int((step_size_min * 60) / sampling_rate_sec)
    
    search_path = os.path.join(raw_data_dir, "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")
        
    print(f"Starting Data Augmentation on {len(csv_files)} test files...")
    all_augmented_dfs = []
    
    for file_path in csv_files:
        try:
            # 1. Feature Engineering
            feat_df = transform_single_csv(file_path, lags=lags, rolling_window=rolling_window, resample_sec=resample_sec)
            
            if feat_df is not None:
                # 2. Windowing / Augmentation
                augmented_df = create_time_windows(
                    feat_df, 
                    window_size_steps=window_size_steps, 
                    step_size_steps=step_size_steps
                )
                if augmented_df is not None:
                    all_augmented_dfs.append(augmented_df)
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            
    if not all_augmented_dfs:
        raise ValueError("Unable to generate the augmented dataset. Check the length of the tests.")
        
    # Vertical concatenation of all fragments from all files
    global_augmented_dataset = pd.concat(all_augmented_dfs, ignore_index=True)
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "dataset_augmented.parquet")
    global_augmented_dataset.to_parquet(output_file_path, index=False)
    
    print(f"Data Augmentation completed successfully!")
    print(f"Resulting global matrix: {global_augmented_dataset.shape[0]} rows, {global_augmented_dataset.shape[1]} columns.")
    print(f"File saved at: {output_file_path}")
    
    return global_augmented_dataset


# ════════════════════════════════════════════════════════════════════════════════
# GENERIC DATA PREPARATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    df,
    group_col="file_name",
    target_class_col="label",
    target_reg_col="C_final",
    exclude_cols=None,
    exclude_patterns=None,
    verbose=True
):
    """
    Generic dataset preparation function that extracts features and targets
    from any augmented dataset, regardless of the number of features.
    
    This function automatically excludes non-numeric columns and specified patterns,
    making it suitable for any dataset structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input augmented dataset (output of build_augmented_dataset).
    
    group_col : str, default="file_name"
        Column name that identifies data groups (used for GroupKFold).
    
    target_class_col : str, default="label"
        Column name for binary classification target.
    
    target_reg_col : str, default="C_final"
        Column name for regression target (continuous value).
    
    exclude_cols : list of str, optional
        Additional column names to exclude (e.g., ['time', 'window_id']).
        If None, defaults to ['time', 'window_id'].
    
    exclude_patterns : list of str, optional
        Column name patterns to exclude (e.g., ['C_final_']).
        Columns containing any of these patterns will be excluded.
    
    verbose : bool, default=True
        If True, prints information about the extracted features and targets.
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix (all numeric columns except targets and metadata).
    
    y_class : pd.Series
        Binary classification target.
    
    y_reg : pd.Series
        Continuous regression target.
    
    groups : pd.Series
        Group identifier column (usually file_name).
    
    Example:
    --------
    >>> X, y_class, y_reg, groups = prepare_dataset(
    ...     df_augmented,
    ...     group_col="file_name",
    ...     target_class_col="label",
    ...     target_reg_col="C_final"
    ... )
    >>> # X now contains 21 numeric features (agnostic to feature names/count)
    >>> # y_class and y_reg contain the targets
    >>> # groups identifies which file each sample comes from
    """
    
    # Set default exclusions
    if exclude_cols is None:
        exclude_cols = ['time', 'window_id']
    else:
        # Ensure basic metadata columns are always excluded
        exclude_cols = list(set(exclude_cols) | {'time', 'window_id'})
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    # Add the target and group columns to the exclusion list
    all_exclude = set(exclude_cols) | {target_class_col, target_reg_col, group_col}
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Apply exclusions: remove excluded column names and patterns
    feature_cols = [
        col for col in numeric_cols
        if col not in all_exclude
        and not any(pattern in col for pattern in exclude_patterns)
    ]
    
    # Extract components
    X = df[feature_cols].copy()
    y_class = df[target_class_col].copy()
    y_reg = df[target_reg_col].copy()
    groups = df[group_col].copy()
    
    if verbose:
        print(f"✓ Dataset Preparation Summary:")
        print(f"  - Features extracted: {len(feature_cols)} numeric columns")
        print(f"  - Feature columns: {feature_cols}")
        print(f"  - Classification target (y_class): '{target_class_col}'")
        print(f"  - Regression target (y_reg): '{target_reg_col}'")
        print(f"  - Group column (GroupKFold): '{group_col}'")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Data shape: X = {X.shape}")
    
    return X, y_class, y_reg, groups