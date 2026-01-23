import numpy as np
import pandas as pd
import json

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


def json_handler(file_path : str, weights_decay : str, loss_type : str):
    """
    Load a json file and modifies its content as a dictionary.

    Params:
    - file_path : string, path to the json file
    - weights_decay : string, type of weights decay to be used in the model
    - loss_type : string, type of loss function to be used in the model

    """

    with open(file_path, 'r') as f:
        config = json.load(f)

    assert weights_decay in ["uni", "soft_lin", "strong_lin", "exp"], f"weights_decay {weights_decay} not recognized. Choose among 'uni', 'soft_lin', 'strong_lin', 'exp'"
    config['weights_decay'] = weights_decay
    assert loss_type in ["horizon_weighted_huber", "mse"], f"loss_type {loss_type} not recognized. Choose among 'horizon_weighted_huber', 'mse'"
    config['loss'] = loss_type

    # save the modified config back to the json file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration file {file_path} modified: weights_decay set to {weights_decay}, loss set to {loss_type}")
    return


def data_definer(all_combinations : bool = True, dataset : str):
    """
    Returns instances of data models. 
    """
    pass


def data_trainer(models : dict, dataset : str):
    """
    Trains data models for a given dataset.
    """
    assert dataset in ['electricity', 'solar', 'traffic', 'volatility', 'wind'], f"Dataset {dataset} not recognized. Choose among 'electricity', 'solar', 'traffic', 'volatility', 'wind'"
    pass


def data_define_and_train(all_combinations : bool = True, dataset : str):
    """
    Defines and trains data models for a given dataset.
    """
    data_definer(all_combinations, dataset)
    data_trainer(models, dataset)
    pass



if __name__ == '__main__':
    pass