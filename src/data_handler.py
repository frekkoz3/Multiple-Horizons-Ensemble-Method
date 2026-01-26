import numpy as np
import pandas as pd
import json

from statsmodels.tsa.statespace.tools import diff
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


def data_filler(time_series: np.ndarray, method: str = 'spline') -> np.ndarray:
    """
    Fill missing NaN values in the time series data.
    Params:
    - time_series : array of shape (n_samples,)
    - method : string, method to fill missing values ('linear', 'cubic', etc.)
    """
    time_series = pd.Series(time_series)
    time_series = time_series.interpolate(method=method).to_numpy()
    return time_series


def data_scaler(time_series: np.ndarray, type : str = 'rob') -> np.ndarray:
    """
    Applies a scaler to the time series data.
    Params:
    - time_series : array of shape (n_samples,)
    - type : string, type of scaler to apply ('rob' for RobustScaler, 'std' for StandardScaler, 'minmax' for MinMaxScaler)
    """
    time_series = time_series.reshape(-1, 1)
    if type == 'rob':
        scaler = RobustScaler()
    elif type == 'std':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    time_series = scaler.fit_transform(time_series).flatten()
    return time_series


def data_differentiator(time_series: np.ndarray) -> np.ndarray:
    """
    Differentiate time series data.
    Params: 
    - time_series : array of shape (n_samples,)
    - diff_order : array of shape (2), differentiation order and seasonal order
    """
    time_series = diff(time_series, k_diff=diff_order[0], k_seasonal_diff=diff_order[1], seasonal_periods=24)
    return 


def data_time_aggregator(time_series: np.ndarray, freq: str) -> np.ndarray:
    """
    Aggregate time series data to a different frequency.
    Params:
    - time_series : array of shape (n_samples,)
    - freq : string, frequency to aggregate to 
    """
    time_series = pd.Series(time_series).resample(freq).mean().to_numpy()

    return time_series


def data_preprocessing(time_series: np.ndarray, aggregator : bool = False, aggregator_window : int = 1, differentiator : bool = False, differentiator_orders : list = [1, 0], scaler : bool = False, scaler_type : str = 'rob', filler : bool = False, filler_type : str = 'splines') -> np.ndarray:
    """
    Do some preprocessing on the time series data.

    Params:
    - time_series : array of shape (n_samples,)
    - aggregator : whether to apply time aggregation
    - aggregator_window : window size for time aggregation
    - differentiator : whether to apply differentiation
    - differentiator_orders : array of shape (2), differentiation order and seasonal order
    - scaler : whether to apply scaling

    Returns:
    - time_series : preprocessed time series
    """ 
    if aggregator:
        time_series = data_time_aggregator(time_series, freq=aggregator_window)
    if differentiator:
        time_series = data_differentiator(time_series, diff_order=differentiator_orders)
    if scaler:
        assert scaler_type in ['rob', 'std', 'minmax'], f"scaler_type {scaler_type} not recognized. Choose among 'rob', 'std', 'minmax'"
        time_series = data_scaler(time_series, scaler_type=scaler_type)
    
    return time_series
   

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


def data_loader(data_path : str, dataset : str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load dataset from csv file and return the whole dataset and the target time series.

    Params:
    - data_path : string, path to the csv file
    - dataset : string, name of the dataset to load. Supported datasets: 'electricity', 'solar', 'traffic', 'volatility', 'wind'

    Returns:
    - time_series : np.ndarray, array of shape (n_samples,) containing the target time series
    - data : pd.DataFrame, dataframe containing the dataset
    """
    with open(data_path + '/data_config.json', 'r') as f:
        config = json.load(f)

    assert dataset in config, f"Dataset {dataset} not found in configuration file data_config.json"

    dataset_path = data_path + '/' + config[dataset]['filename']
    id_col = config[dataset]['id_col']
    date_col = config[dataset]['date_col']
    target_col = config[dataset]['target_col']
    id_target = config[dataset]['id_target']

    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    try:
        target_series = data[data[id_col] == id_target].sort_values(date_col).set_index(date_col)
        time_series = target_series[target_col]
    except KeyError as e:
        print(f"Error: {e}")
        raise

    return time_series , data


def json_handler(file_path : str, weights_decay : str, loss_type : str, horizon : int, window : int):
    """
    Load a json file and modifies its content as a dictionary.

    Params:
    - file_path : string, path to the json file
    - weights_decay : string, type of weights decay to be used in the model
    - loss_type : string, type of loss function to be used in the model
    - horizon : integer, the length of the horizon forecasting
    - window : integer, the length of the window for predictions

    """

    with open(file_path, 'r') as f:
        config = json.load(f)

    assert weights_decay in ["uni", "soft_lin", "strong_lin", "exp"], f"weights_decay {weights_decay} not recognized. Choose among 'uni', 'soft_lin', 'strong_lin', 'exp'"
    config['weights_decay'] = weights_decay
    assert loss_type in ["horizon_weighted_huber", "mse"], f"loss_type {loss_type} not recognized. Choose among 'horizon_weighted_huber', 'mse'"
    config['loss'] = loss_type

    config['horizon'] = horizon
    config['window'] = window

    # save the modified config back to the json file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration file {file_path} modified: weights_decay set to {weights_decay}, loss set to {loss_type}")
    return


def dataset_handler(dataset_init : str, data_path : str, prop : tuple = (0.6, 0.2, 0.2), shuffle : bool = False, random_state : int | None = None, **preprocess_kwargs) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Handles data loading, preprocessing, and splitting for a given dataset.
    
    Params:
    - dataset_init : string, initials of the dataset to load
    - data_path : string, path to the configuration file
    - prop : tuple, proportions for train/val/test split
    - shuffle : boolean, whether to shuffle data before splitting
    - random_state : integer or None, random state for reproducibility if shuffle is True

    **preprocess_kwargs : keyword arguments for data_preprocessing function:
        - window : length of input window for sliding window
        - horizon : length of output horizon for sliding window
        - aggregator : whether to apply time aggregation
        - aggregator_window : window size for time aggregation
        - differentiator : whether to apply differentiation
        - differentiator_orders : array of shape (2), differentiation order and seasonal order
        - scaler : whether to apply scaling
        - scaler_type : type of scaler to apply ('rob' for RobustScaler, 'std' for StandardScaler, 'minmax' for MinMaxScaler)


    Returns:
    - train : tuple, training set (X_train, y_train)
    - val : tuple, validation set (X_val, y_val)
    - test : tuple, test set (X_test, y_test)
    """ 
    # Load data
    X, data = data_loader(data_path = data_path, dataset = dataset_initials[dataset_init])
    # Preprocess data
    X = data_preprocessing(X, *preprocess_kwargs)

    # Create sliding windows
    window = preprocess_kwargs.get('window', 48)
    horizon = preprocess_kwargs.get('horizon', 12)

    X_slide, y_slide = sliding_window(X, window=window, horizon=horizon)
    # Split data
    train, val, test = train_validation_test_split(X_slide, y_slide, prop=prop, shuffle=shuffle, random_state=random_state)  

    # return train, val, test
    return train, val, test
    


def models_definer(dataset_init : list[str] | str, operation : list[str], config_model_path : str, **kwargs) -> dict:
    """
    Defines data models for a given dataset.

    Params:
    - dataset_init : string or list of strings, initial letter(s) of the dataset. Supported: 'e' for electricity, 's' for solar, 't' for traffic, 'v' for volatility, 'w' for wind
    - operation : list of strings, operations to perform. Supported operations: 'losses', 'models', 'weights' and their combinations
    - config_model_path : string, path to the model configuration file
    - **kwargs : keyword arguments for model characteristics non-specific for the training phase

    Returns:
    - models : dictionary of models defined for the dataset. For many datasets, returns a list of dictionaries. Dictionaries are always return in the same order of dataset_init 

    Notes:
    `operation` defines the kind of combinations of models to cycle on for the model definition. 
    
    In particular, should be always defined in `**kwargs` :
    1. loss_type: (list of) type(s) of loss function to use. Supported: 'horizon_weighted_huber', 'mse'
    2. model_type: (list of) type(s) of model to use. Supported: 'TCN', 'XGBoost', 'ARIMA', 'UHMEMe'
    3. weights_type: (list of) type(s) of weighting strategy to use. Supported: 'uni', 'soft_lin', 'strong_lin', 'exp'
    """
    assert dataset_init in ['e', 's', 't', 'v', 'w'], f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    
    models_for_each_dataset = []
    for ds in dataset_init if isinstance(dataset_init, list) else [dataset_init]:
        models = {}
        # TODO
        models_for_each_dataset.append(models)      
    
    if len(models_for_each_dataset) == 1:
        return models_for_each_dataset[0]

    return models_for_each_dataset

    

def models_trainer(models : dict, train : np.ndarray, dataset_init : str, save_models : bool = True):
    """
    Trains data models for a given dataset.

    Params:
    - models : dictionary of models to train
    - train : string, name of the dataset to use
    - dataset_init : string, initials of the dataset to load
    - save_models : boolean, whether to save the trained models
    """

    for model in models:
        print(f"\n\nTrain model: {model}")
        model.fit(train[0], train[1])

        # if model is UMHEME, compute weights
        if str(model).__contains__("UMHEMe"):
            model.compute_weights(train[0], train[1])

        if save_models:
            model.save_model(f"../models/{str(model)}_{dataset_init}.model")

    return 

    
def models_evaluator(models : dict, test : list[np.ndarray] | None):
    """
    Evaluates data models for a given dataset.

    Params:
    - models : dictionary of fitted models to evaluate
    - test : tuple, test set (X_test, y_test)
    """
    results = {}
    print("\n\nTest set evaluation:")
    for model in models:
        print(f"\nEvaluate model: {model}")
        if str(model).__contains__("UMHEMe"):
            results[str(model)] = (model.predict(test[0]), model.whole_predict(test[0]))
        else:
            results[str(model)] = model.predict(test[0])

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