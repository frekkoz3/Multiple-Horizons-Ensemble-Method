import numpy as np

def train_validation_test_split(X : np.ndarray, y : np.ndarray, proportions : tuple = (0.6, 0.2, 0.2), shuffle : bool = False, random_state: int | None = None):
    """
        Split a temporal dataset into train, validation and test set. No shuffle

        Params :
        - X : set of inputs of shape number_of_series x window
        - y : set of ground truth of shape number_of_series x horizon
        - proportion : tuple containing the porportion of data to be stored in the train set, in the validation set, in the test set
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

def sliding_window(time_series: np.ndarray, window: int, horizon: int, k: int = 1):
    """
    Split a 1D time series into sliding windows of size window + horizon.
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

if __name__ == '__main__':
    pass