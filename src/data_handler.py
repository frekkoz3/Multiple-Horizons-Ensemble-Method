import numpy as np
import pandas as pd
import json
import os

from src.direct_models import *
from src.mheme import *
from src.config_files import *

from statsmodels.tsa.statespace.tools import diff
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
   

def retrieve_data_day_from_index(time_series :  np.ndarray, index :  int, data_path : str, proportions : tuple=(0.6, 0.2, 0.2), shuffled : bool = False, random_state : int | None = None) -> str :
    """
    Given a series and an index of the series among the whole dataset, returns the day (and possibly the hour) of the first element of the series.

    Params:
    - time_series : array of shape (window, ) or (horizon, )
    - index : integer, index of the time_series in the dataset it belongs (train, validation, test)
    - dataset : string, the file path to the dataset used.
    - proportions : tuple of float, the proportions of train/val/test split used
    - shuffled : boolean, True if during train_validation_test_split data were shuffled
    - random_state : integer or None, random_state used for shuffling. Must be integer if shuffled is True

    Returns:
    - day : day and hour of the first element of the dataset
    """

    if shuffled:
        assert random_state is not None, "if data are shuffled, is impossible to retrieve data day"
        # TODO
        raise NotImplementedError("Retrieving data day from index is not implemented for shuffled data.")

    dataset = pd.read_csv(data_path)
    n_samples = dataset.shape[0]

    p_train, p_val, p_test = proportions
    n_train = int(n_samples * p_train)
    n_val   = int(n_samples * p_val)

    if index < n_train:
        real_index = index
    elif index < n_train + n_val:
        real_index = index + n_train
    else:
        real_index = index + n_train + n_val

    if 'traffic.csv' in data_path:    # if dataset 'traffic.csv', the target column is 'hours_from_start'
        hours_from_start = dataset.iloc[real_index]['hours_from_start']
        day = f"Day {hours_from_start // 24}, Hour {hours_from_start % 24}"
    else:                              # else, other datasets have 'date' column as target
        day = dataset.iloc[real_index]['date']

    return day


def json_handler(file_path : str, weights_decay : str | None = None, loss_type : str | None = None, horizon : int | None = None, window : int | None = None, id_target : str | int | None = None, skip : int | None = None ,dataset : str | None = None) -> None:
    """
    Load a json file and modifies its content as a dictionary.

    Params:
    - file_path : str, path to the json file
    - weights_decay : str, type of weights decay to be used in the model
    - loss_type : str, type of loss function to be used in the model
    - horizon : int, the length of the horizon forecasting
    - window : int, the length of the window for predictions
    - id_target : str or int, the target id value to filter the dataset

    """

    with open(file_path, 'r') as f:
        config = json.load(f)

    if weights_decay is not None:
        assert weights_decay in ["uni", "soft_lin", "strong_lin", "exp"], f"weights_decay {weights_decay} not recognized. Choose among 'uni', 'soft_lin', 'strong_lin', 'exp'"
        config['weights_decay'] = weights_decay
    if loss_type is not None:
        assert loss_type in ["horizon_weighted_huber", "mse"], f"loss_type {loss_type} not recognized. Choose among 'horizon_weighted_huber', 'mse'"
        config['loss'] = loss_type

    if id_target is not None:
        assert dataset is not None, "if id_target is specified, dataset must be specified too"
        config[dataset]['id_target'] = id_target

    if horizon is not None:
        config['horizon'] = horizon
    if window is not None:
        config['window'] = window

    if skip is not None:
        config['skip'] = skip

    # save the modified config back to the json file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration file {file_path} modified")
    return


def data_filler(time_series: np.ndarray, method: str = 'spline') -> np.ndarray:
    """
    Fill missing NaN values in the time series data.
    Params:
    - time_series : array of shape (n_samples,)
    - method : string, method to fill missing values ('linear', 'cubic', 'spline'). Default 'spline'
    """
    assert method in ['linear', 'cubic', 'spline'], f"method {method} not recognized. Choose among 'linear', 'cubic', 'spline'"
    time_series = pd.Series(time_series)
    time_series = time_series.interpolate(method=method).to_numpy()
    return time_series


def data_scaler(time_series: np.ndarray, scaler_type : str = 'rob') -> np.ndarray:
    """
    Applies a scaler to the time series data.
    Params:
    - time_series : array of shape (n_samples,)
    - type : string, type of scaler to apply ('rob' for RobustScaler, 'std' for StandardScaler, 'minmax' for MinMaxScaler)
    """
    assert scaler_type in ['rob', 'std', 'minmax'], f"type {scaler_type} not recognized. Choose among 'rob', 'std', 'minmax'"

    was_1d = False
    if hasattr(time_series, 'ndim') and time_series.ndim == 1:
        was_1d = True
        # If is pd.Series use .values.reshape. If is np.ndarray use .reshape
        values_2d = time_series.values.reshape(-1, 1) if hasattr(time_series, 'values') else time_series.reshape(-1, 1)
    else:
        # Is already 2D (e.g., is a multivariate matrix)
        values_2d = time_series
    
    if scaler_type == 'rob':
        scaler = RobustScaler()
    elif scaler_type == 'std':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(values_2d)

    if was_1d:
        return scaled_data.flatten()
    else:
        return scaled_data
        

def data_differentiator(time_series: np.ndarray, diff_order: list[int]) -> np.ndarray:
    """
    Differentiate time series data.
    Params: 
    - time_series : array of shape (n_samples,)
    - diff_order : array of shape (3), differentiation order, seasonal order, and seasonal periods
    """
    assert len(diff_order) == 3, "diff_order must be of length 3"
    time_series = diff(time_series, k_diff=diff_order[0], k_seasonal_diff=diff_order[1], seasonal_periods=diff_order[2])
    return time_series


def data_time_aggregator(time_series: np.ndarray, freq: str | int) -> np.ndarray:
    """
    Aggregate time series data to a different frequency.
    Handles both 1D arrays and 2D matrices (Time x UIDs).
    """
    
    # 1. Gestione Matrice 2D (Multi-UID) vs 1D
    if time_series.ndim == 2:
        # Se è 2D (es. 52560 righe, 100 colonne), usiamo DataFrame.
        # Ogni colonna rappresenta un UID diverso.
        df = pd.DataFrame(time_series)
    else:
        # Se è 1D, usiamo Series
        df = pd.Series(time_series)

    # 2. Gestione della Frequenza
    if isinstance(freq, int):
        # CASO A: Frequenza numerica (es. freq=4 significa media ogni 4 righe)
        # Raggruppa per indice intero
        time_series_agg = df.groupby(df.index // freq).mean().to_numpy()
        
    elif isinstance(freq, str):
        # CASO B: Frequenza stringa (es. '1H')
        # .resample() richiede un indice temporale. Poiché time_series è un numpy array
        # senza date, dobbiamo creare un indice fittizio per farlo funzionare.
        
        # Assumiamo una data di partenza arbitraria e una frequenza base (es. 15min o 10min)
        # Se il tuo dataset originale è ogni 15 minuti ('15T'), mettilo qui:
        base_freq = '15T'  
        
        df.index = pd.date_range(start="2000-01-01", periods=len(df), freq=base_freq)
        
        # Ora resample funzionerà su tutte le colonne (UID) contemporaneamente
        time_series_agg = df.resample(freq).mean().to_numpy()
        
    else:
        raise ValueError(f"Frequenza {freq} non supportata. Usa int o str.")

    return time_series_agg


def train_validation_test_split(X : np.ndarray, y : np.ndarray, proportions : tuple = (0.6, 0.2, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state: int | None = None) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
        Split a temporal dataset into train, validation, and test set. No shuffle

        Params :
        - X : set of inputs of shape number_of_series x window
        - y : set of ground truth of shape number_of_series x horizon
        - proportion : tuple containing the proportion of data to be stored in the train set, in the validation set, in the test set
        - shuffle_data : whether to shuffle samples before splitting
        - shuffle_internal : whether to shuffle samples within each split
        - random_state : seed for reproducibility (used if shuffle=True)

        Return :
        - (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    assert len(proportions) == 3, "proportions must have length 3"

    p_train, p_val, p_test = proportions
    assert np.isclose(p_train + p_val + p_test, 1.0), "proportions must sum to 1"

    n = X.shape[0]

    if shuffle_data:
        assert random_state is not None, "if data are shuffled before splitting, random_state must be initialized"
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(n)
        X = X[idx]
        y = y[idx]

    n_train = int(n * p_train)
    n_val   = int(n * p_val)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    if shuffle_internal:
        assert random_state is not None, "if data are shuffled internally of the splits, random_state must be initialized"
        rng = np.random.default_rng(random_state)

        idx_train = rng.permutation(X_train.shape[0])
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        idx_val = rng.permutation(X_val.shape[0])
        X_val = X_val[idx_val]
        y_val = y_val[idx_val]

        idx_test = rng.permutation(X_test.shape[0])
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]


    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def sliding_window(time_series: np.ndarray, window: int, horizon: int, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 1D time series into sliding windows of size (window + horizon).
    Each window produces (X, y) with length `window` and `horizon` respectively.
    Windows are separated by lag `k`.

    Params:
    - time_series : array of shape (n_samples,)
    - window : length of input window
    - horizon : length of output horizon
    - k : step between sliding windows

    Returns:
    - X : array of shape (n_windows, window)
    - y : array of shape (n_windows, horizon)
    """
    time_series = np.asarray(time_series)

    # CASE 1: 1D Time series
    if time_series.ndim == 1:
        n_samples = len(time_series)
        n_windows = (n_samples - window - horizon) // k + 1
        
        if n_windows <= 0:
            raise ValueError("Time series too short for the given window and horizon.")
    
        X = np.zeros((n_windows, window))
        y = np.zeros((n_windows, horizon))
    
        for i in range(n_windows):
            start = i * k
            X[i] = time_series[start : start + window]
            y[i] = time_series[start + window : start + window + horizon]
    
        return X, y

    elif time_series.ndim == 2:
        n_ids, n_samples = time_series.shape
        X_list = []
        y_list = []
        
        # Iterate over each ID (row)
        for i in range(n_ids):
            single_series = time_series[i, :]
            
            # Calculate number of windows for this specific series
            n_windows = (n_samples - window - horizon) // k + 1
            
            if n_windows > 0:
                # Use the logic from Case 1 for this row
                X_i = np.zeros((n_windows, window))
                y_i = np.zeros((n_windows, horizon))
                
                valid_windows = True
                
                for j in range(n_windows):
                    start = j * k
                    X_i[j] = single_series[start : start + window]
                    y_i[j] = single_series[start + window : start + window + horizon]
                
                X_list.append(X_i)
                y_list.append(y_i)
        
        # Vertically stack all results
        if not X_list:
            raise ValueError("No valid windows generated from the provided matrix.")
            
        X_total = np.vstack(X_list)
        y_total = np.vstack(y_list)
        
        return X_total, y_total
    
    else:
        raise ValueError("time_series must be 1D or 2D array.")


def data_preprocessing(time_series: np.ndarray, data_config_path : str, dataset_init : str, aggregator : bool = False, aggregator_window : int = 1, differentiator : bool = False, differentiator_orders : list = [1, 0, 24], scaler : bool = False, scaler_type : str = 'rob', filler : bool = False, filler_type : str = 'splines') -> np.ndarray:
    """
    Do some preprocessing on the time series data.

    Params:
    - time_series : array of shape (n_samples,)
    - data_config_path : string, path to the data configuration file
    - dataset_init : string, initials of the dataset to load. Supported initials: 'e' for electricity, 's' for solar, 't' for traffic, 'v' for volatility, 'w' for wind

    - aggregator : whether to apply time aggregation
    - aggregator_window : window size for time aggregation
    - differentiator : whether to apply differentiation
    - differentiator_orders : array of shape (3), differentiation order, seasonal order, and seasonal periods
    - scaler : whether to apply scaling
    - scaler_type : type of scaler to apply ('rob' for RobustScaler, 'std' for StandardScaler, 'minmax' for MinMaxScaler)
    - filler : whether to apply missing value filling
    - filler_type : method to fill missing values ('linear', 'cubic', 'spline')

    Returns:
    - time_series : preprocessed time series
    """ 
    with open(data_config_path, 'r') as f:
        config = json.load(f)
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    dataset = datasets[dataset_init]
    try:
        if config[dataset]['aggregator']:
            time_series = data_time_aggregator(time_series, freq=config[dataset]['aggregator_window'])
        if config[dataset]['differentiator']:
            time_series = data_differentiator(time_series, diff_order= config[dataset]['differentiator_orders'])
        if config[dataset]['scaler']:
            assert config[dataset]['scaler_type'] in ['rob', 'std', 'minmax'], f"scaler_type {config[dataset]['scaler_type']} not recognized. Choose among 'rob', 'std', 'minmax'"
            time_series = data_scaler(time_series, scaler_type=config[dataset]['scaler_type'])
        if config[dataset]['filler']:
            assert config[dataset]['filler_type'] in ['linear', 'cubic', 'spline'], f"filler_type {config[dataset]['filler_type']} not recognized. Choose among 'linear', 'cubic', 'spline'"
            time_series = data_filler(time_series, method=config[dataset]['filler_type'])
    except KeyError as e:
        print(f"Error: {e}")
        raise
    return time_series


def data_loader(data_path : str, data_config_path : str, dataset_init : str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load dataset from csv file and return the whole dataset and the target time series.

    Params:
    - data_path : string, path to the csv file
    - data_config_path : string, path to the data configuration file
    - dataset_init : string, initial letter of the dataset to load. Supported initials: 'e' for electricity, 's' for solar, 't' for traffic, 'v' for volatility, 'w' for wind

    Returns:
    - time_series : np.ndarray, array of shape (n_samples,) containing the target time series
    - data : pd.DataFrame, dataframe containing the dataset

    Notes:
    if value of id_target in data_config_path is 'ALL', the function returns all the time series in a pivoted format (rows: ids, columns: time steps)
    """
    with open(data_config_path, 'r') as f:
        config = json.load(f)

    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    dataset = datasets[dataset_init]

    dataset_path = data_path + '/' + config[dataset]['filename']
    id_col = config[dataset]['id_col']
    date_col = config[dataset]['date_col']
    date_format = config[dataset]['date_format']
    target_col = config[dataset]['target_col']
    id_target = config[dataset]['id_target']

    try:
        data = pd.read_csv(dataset_path, parse_dates=[date_col], date_format=date_format)    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    try:
        if id_target != 'ALL': # We are working on a specific series
            target_series = data[data[id_col] == id_target].sort_values(date_col).set_index(date_col)
            time_series = target_series[target_col]

            return time_series , data

        else:  # We are working on all the time series, so we aggregate them
            multiple_time_series = data.pivot(index=date_col, columns=id_col, values=target_col).sort_index()
            return multiple_time_series , data
    except KeyError as e:
        print(f"Error: {e}")
        raise


def dataset_handler(dataset_init : str, data_path : str, data_config_path : str, is_arima : bool = False, prop : tuple = (0.6, 0.2, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int | None = None) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Handles data loading, preprocessing, and splitting for a given dataset.
    
    Params:
    - dataset_init : string, initials of the dataset to load
    - data_path : string, path to the configuration file
    - data_config_path : string, path to the data configuration file
    - is_arima : boolean, whether the model to be used is ARIMA
    - prop : tuple, proportions for train/val/test split
    - shuffle_data : boolean, whether to shuffle data before splitting
    - shuffle_internal : boolean, whether to shuffle data after splitting
    - random_state : integer or None, random state for reproducibility if shuffle is True

    Returns:
    - train : tuple, training set (X_train, y_train)
    - val : tuple, validation set (X_val, y_val)
    - test : tuple, test set (X_test, y_test)
    - X : np.ndarray, preprocessed time series used for training/testing
    - data : pd.DataFrame, original dataframe loaded from csv

    Notes:
    data_config_path must contain the following preprocessing parameters for each dataset:
        - window : length of input window for sliding window. Default 48
        - horizon : length of output horizon for sliding window. Default 12

        - aggregator : whether to apply time aggregation
        - aggregator_window : window size for time aggregation
        - differentiator : whether to apply differentiation
        - differentiator_orders : array of shape (2), differentiation order and seasonal order
        - scaler : whether to apply scaling
        - scaler_type : type of scaler to apply ('rob' for RobustScaler, 'std' for StandardScaler, 'minmax' for MinMaxScaler)
        - filler : whether to apply missing value filling
        - filler_type : method to fill missing values ('linear', 'cubic', 'spline')

    ARIMA models work differently, as they do not need sliding windows. In this case, the function returns:
        - train : tuple, training set (X_train, None) where X_train is the whole series used for training
        - val : None    
        - test : tuple, test set (X_test, None) where X_test is the whole series used for testing
        - X : np.ndarray, preprocessed time series used for training/testing
        - data : pd.DataFrame, original dataframe loaded from csv
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    dataset = datasets[dataset_init]

    # Load data
    X, data = data_loader(data_path = data_path, data_config_path = data_config_path, dataset_init = dataset_init)
    # Preprocess data
    X = data_preprocessing(X, data_config_path = data_config_path, dataset_init = dataset_init)
    # if X is 2D (Multiple series) is currently (Time, IDs)
    if X.ndim == 2:
        X = X.T
    # Create sliding windows
    with open(data_config_path, 'r') as f:
        config = json.load(f)
    
    if not is_arima:
        X_slide, y_slide = sliding_window(X, window=config[dataset]['window'], horizon=config[dataset]['horizon'], k=config[dataset]['skip'])
        # Split data
        train, val, test = train_validation_test_split(X_slide, 
                                                       y_slide, 
                                                       proportions=prop, 
                                                       shuffle_data=shuffle_data, 
                                                       shuffle_internal=shuffle_internal, 
                                                       random_state=random_state)  

        # return train, val, test
        return train, val, test, X, data

    else: # we are working with ARIMA. We do not want sliding windows, but the whole series with size prop[0]+prop[1] as "train" and prop[2] as "test"
        n_samples = X.shape[0]
        n_train = int(n_samples * (prop[0] + prop[1]))
        X_train = X[:n_train]
        X_test = X[n_train:]
        return (X_train, None), None, (X_test, None), X, data
    

if __name__ == '__main__':
    # Global variables
    WINDOW = 48
    HORIZON = 12
    
    DATA_PATH = '../data'
    
    TCN_PATH_CONFIG_LOAD = '../src/config_files/tcn_config.json'
    TCN_PATH_SAVE = '../models/tcn'
    
    XGB_PATH_CONFIG_LOAD = '../src/config_files/xgb_config.json'
    XGB_PATH_SAVE = '../models/xgb'


    # Load data
    dataset_initials = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    
    for ds in dataset_initials:
        print(f"Loading dataset: {dataset_initials[ds]}")
        X, data = data_loader(data_path = DATA_PATH, dataset = dataset_initials[ds])
    
        # Change name of X, data based on ds (This is pure flex):
        globals()[f'X_{ds}'] = X
        globals()[f'data_{ds}'] = data