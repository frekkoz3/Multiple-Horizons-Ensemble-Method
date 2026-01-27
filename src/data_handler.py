import numpy as np
import pandas as pd
import json
import os

from src.direct_models import *
from src.mheme import *
from src.config_files import *

from statsmodels.tsa.statespace.tools import diff
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
   

def train_validation_test_split(X : np.ndarray, y : np.ndarray, proportions : tuple = (0.6, 0.2, 0.2), shuffle : bool = False, random_state: int | None = None) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
        Split a temporal dataset into train, validation, and test set. No shuffle

        Params :
        - X : set of inputs of shape number_of_series x window
        - y : set of ground truth of shape number_of_series x horizon
        - proportion : tuple containing the proportion of data to be stored in the train set, in the validation set, in the test set
        - shuffle : whether to shuffle samples before splitting
        - random_state : seed for reproducibility (used if shuffle=True)

        Return :
        - (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    assert len(proportions) == 3, "proportions must have length 3"

    p_train, p_val, p_test = proportions
    assert np.isclose(p_train + p_val + p_test, 1.0), "proportions must sum to 1"

    n = X.shape[0]

    if shuffle:
        assert random_state is not None, "if data are shuffled, random_state must be initialized"
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

    print(real_index)

    if 'traffic.csv' in data_path:    # if dataset 'traffic.csv', the target column is 'hours_from_start'
        hours_from_start = dataset.iloc[real_index]['hours_from_start']
        day = f"Day {hours_from_start // 24}, Hour {hours_from_start % 24}"
    else:                              # else, other datasets have 'date' column as target
        day = dataset.iloc[real_index]['date']

    return day


def json_handler(file_path : str, weights_decay : str | None = None, loss_type : str | None = None, horizon : int | None = None, window : int | None = None):
    """
    Load a json file and modifies its content as a dictionary.

    Params:
    - file_path : str, path to the json file
    - weights_decay : str, type of weights decay to be used in the model
    - loss_type : str, type of loss function to be used in the model
    - horizon : int, the length of the horizon forecasting
    - window : int, the length of the window for predictions

    """

    with open(file_path, 'r') as f:
        config = json.load(f)

    assert weights_decay in ["uni", "soft_lin", "strong_lin", "exp"], f"weights_decay {weights_decay} not recognized. Choose among 'uni', 'soft_lin', 'strong_lin', 'exp'"
    if weights_decay is not None:
        config['weights_decay'] = weights_decay
    if loss_type is not None:
        assert loss_type in ["horizon_weighted_huber", "mse"], f"loss_type {loss_type} not recognized. Choose among 'horizon_weighted_huber', 'mse'"
        config['loss'] = loss_type

    if horizon is not None:
        config['horizon'] = horizon
    if window is not None:
        config['window'] = window

    # save the modified config back to the json file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration file {file_path} modified")
    return


def dataset_handler(dataset_init : str, data_path : str, data_config_path : str, prop : tuple = (0.6, 0.2, 0.2), shuffle : bool = False, random_state : int | None = None) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Handles data loading, preprocessing, and splitting for a given dataset.
    
    Params:
    - dataset_init : string, initials of the dataset to load
    - data_path : string, path to the configuration file
    - data_config_path : string, path to the data configuration file
    - prop : tuple, proportions for train/val/test split
    - shuffle : boolean, whether to shuffle data before splitting
    - random_state : integer or None, random state for reproducibility if shuffle is True

    Returns:
    - train : tuple, training set (X_train, y_train)
    - val : tuple, validation set (X_val, y_val)
    - test : tuple, test set (X_test, y_test)

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
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    dataset = datasets[dataset_init]

    # Load data
    X, data = data_loader(data_path = data_path, data_config_path = data_config_path, dataset_init = dataset_init)
    # Preprocess data
    X = data_preprocessing(X, data_config_path = data_config_path, dataset_init = dataset_init)

    # Create sliding windows
    with open(data_config_path, 'r') as f:
        config = json.load(f)
    
    X_slide, y_slide = sliding_window(X, window=config[dataset]['window'], horizon=config[dataset]['horizon'])
    # Split data
    train, val, test = train_validation_test_split(X_slide, y_slide, proportions=prop, shuffle=shuffle, random_state=random_state)  

    # return train, val, test
    return train, val, test, X, data


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


def data_time_aggregator(time_series: np.ndarray, freq: str) -> np.ndarray:
    """
    Aggregate time series data to a different frequency.
    Params:
    - time_series : array of shape (n_samples,)
    - freq : string, frequency to aggregate to 
    """
    time_series = pd.Series(time_series).resample(freq).mean().to_numpy()

    return time_series


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
    target_col = config[dataset]['target_col']
    id_target = config[dataset]['id_target']

    try:
        data = pd.read_csv(dataset_path, parse_dates=[date_col])    
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

    

def models_definer(dataset_init : list[str], loss_type : list[str], model_type : list[str], weight_type : list[str], config_model_paths : list[str],  **kwargs) -> dict:
    """
    Defines data models for a given dataset.

    Params:
    - dataset_init : str or list of str, initial letter(s) of the dataset. Supported: 'e' for electricity, 's' for solar, 't' for traffic, 'v' for volatility, 'w' for wind
    - loss_type : list of str, type(s) of loss function to use. Supported: 'horizon_weighted_huber', 'mse'
    - model_type : list of str, type(s) of model to use. Supported: 'TCN', 'XGBoost', 'ARIMA', 'UHMEMe'
    - weight_type : list of str, type(s) of weighting strategy to use. Supported: 'uni', 'soft_lin', 'strong_lin', 'exp'
    - config_model_paths : list of str, path(s) to the model configuration file(s)
    - **kwargs : keyword arguments for UMHEMe characteristics
        - base_model : str, base model for UMHEMe. Supported: 'TCN', 'XGBoost'
        - skip : int, skip parameter for UMHEMe. Default 1

    Returns:
    - models : dictionary of models defined for the dataset. For many datasets, returns a list of dictionaries. Dictionaries are always return in the same order of dataset_init     
    """
    assert all(dataset in ['e', 's', 't', 'v', 'w'] for dataset in dataset_init), f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    assert all(loss in ['horizon_weighted_huber', 'mse'] for loss in loss_type), f"Loss type {loss_type} not recognized. Choose among 'horizon_weighted_huber', 'mse'"
    assert all(model in ['TCN', 'XGBoost', 'ARIMA', 'UMHEMe'] for model in model_type), f"Model type {model_type} not recognized. Choose among 'TCN', 'XGBoost', 'ARIMA', 'UMHEMe'"
    assert all(weight in ['uni', 'soft_lin', 'strong_lin', 'exp'] for weight in weight_type), f"Weights type {weight_type} not recognized. Choose among 'uni', 'soft_lin', 'strong_lin', 'exp'"
    
    models_for_each_dataset = []

    for i, ds in enumerate(dataset_init):
        models = {}

        for loss in loss_type:
            for j, model in enumerate(model_type):
                for weights in weight_type:
                    # Modify json config file
                    json_handler(file_path = config_model_paths[j], weights_decay = weights, loss_type = loss)
                    
                    # Define model
                    if model == 'TCN':
                        models[f'TCN_{loss}_{weights}'] = TCN(file_path = config_model_paths[j])
                    elif model == 'XGBoost':
                        models[f'XGBoost_{loss}_{weights}'] = XGBoost(file_path = config_model_paths[j])
                    elif model == 'ARIMA':
                        models[f'ARIMA_{loss}_{weights}'] = ARIMA(file_path = config_model_paths[j])
                    else:  # UMHEMe
                        base_model_class = None
                        if 'base_model' in kwargs:
                            if kwargs['base_model'] == 'TCN':
                                base_model_class = TCN
                            else:  # XGBoost
                                base_model_class = XGBoost

                            # Need of horizon and window from CONFIG_MODEL_PATH:
                            with open(config_model_paths[j], 'r') as f:
                                config = json.load(f)

                            dataset_window = config['window']
                            dataset_horizon = config['horizon']

                            models[f'UMHEMe_{loss}_{weights}'] = UMHEMe(horizon = dataset_horizon, window = dataset_window, model_class = base_model_class, config_path = config_model_paths[j], skip = kwargs.get('skip', 1))
                        
        models_for_each_dataset.append(models)      
    
    if len(models_for_each_dataset) == 1:
        return models_for_each_dataset[0]

    return models_for_each_dataset

    
def models_trainer(models : dict, train : np.ndarray, dataset_init : str, save_models : bool = True, models_path_save : str = '../models/'):
    """
    Trains data models for a given dataset.

    Params:
    - models : dictionary of models to train
    - train : string, name of the dataset to use
    - dataset_init : string, initials of the dataset to load
    - save_models : boolean, whether to save the trained models
    - models_path_save : string, path to save the trained models
    """

    for model_name, model_instance in models.items():
        
        print(f"\n\nTrain model: {model_name}")
        model_instance.fit(train[0], train[1])

        # if model is UMHEME, compute weights
        if "UMHEMe" in model_name:
            model_instance.compute_weights(train[0], train[1])

        if save_models:
            os.makedirs(models_path_save, exist_ok=True)
            model_instance.save_model(f"{models_path_save}/{model_name}_{dataset_init}.pkl")

    return 

    
def models_evaluator(models : dict, test : list[np.ndarray], dataset_init: str):
    """
    Evaluates data models for a given dataset.

    Params:
    - models : dictionary of fitted models to evaluate
    - test : tuple, test set (X_test, y_test)ù
    - dataset_init : str, ini
    """
    results = {}
    print("\n\nTest set evaluation:")
    for model_name, model_instance in models.items():
        print(f"\nEvaluate model: {model_name}")
        
        if "UMHEMe" in model_name:
            results[model_name] = (model_instance.predict(test[0]), model_instance.whole_predict(test[0]))
        else:
            results[model_name] = model_instance.predict(test[0])

    return results
    

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