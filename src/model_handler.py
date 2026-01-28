import numpy as np
import pandas as pd
import json
import os

from src.direct_models import *
from src.mheme import *
from src.config_files import *
from src.data_handler import *

from statsmodels.tsa.statespace.tools import diff
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def models_definer(dataset_init : list[str], loss_type : list[str], model_type : list[str], weight_type : list[str], data_config_path : str, model_config_paths : list[str],  **kwargs) -> dict:
    """
    Defines data models for a given dataset.

    Params:
    - dataset_init : str or list of str, initial letter(s) of the dataset. Supported: 'e' for electricity, 's' for solar, 't' for traffic, 'v' for volatility, 'w' for wind
    - loss_type : list of str, type(s) of loss function to use. Supported: 'horizon_weighted_huber', 'mse'
    - model_type : list of str, type(s) of model to use. Supported: 'TCN', 'XGBoost', 'ARIMA', 'UHMEMe'
    - weight_type : list of str, type(s) of weighting strategy to use. Supported: 'uni', 'soft_lin', 'strong_lin', 'exp'
    - data_config_path : str, path to the dataset configuration file
    - model_config_paths : list of str, path(s) to the model configuration file(s)
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
    models_names = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    
    
    models_for_each_dataset = []

    for i, ds in enumerate(dataset_init):
        models = {}

        for loss in loss_type:
            for j, model in enumerate(model_type):
                for weights in weight_type:
                    # Modify json config file accordingly to the respective information provided
                    with open(data_config_path, 'r') as f:
                        data_config = json.load(f)
                        
                    json_handler(file_path = model_config_paths[j], weights_decay = weights, loss_type = loss, horizon = data_config[models_names[ds]]['horizon'], window = data_config[models_names[ds]]['window'])
                    
                    # Define model
                    if model == 'TCN':
                        models[f'TCN_{loss}_{weights}'] = TCN(file_path = model_config_paths[j])
                    elif model == 'XGBoost':
                        models[f'XGBoost_{loss}_{weights}'] = XGBoost(file_path = model_config_paths[j])
                    elif model == 'ARIMA':
                        models[f'ARIMA_{loss}_{weights}'] = ARIMA(file_path = model_config_paths[j])
                    else:  # UMHEMe
                        base_model_class = None
                        if 'base_model' in kwargs:
                            if kwargs['base_model'] == 'TCN':
                                base_model_class = TCN
                            else:  # XGBoost
                                base_model_class = XGBoost

                            # Need of horizon and window from CONFIG_MODEL_PATH:
                            with open(model_config_paths[j], 'r') as f:
                                config = json.load(f)

                            dataset_window = config['window']
                            dataset_horizon = config['horizon']

                            models[f'UMHEMe_{loss}_{weights}'] = UMHEMe(horizon = dataset_horizon, window = dataset_window, model_class = base_model_class, config_path = model_config_paths[j], skip = kwargs.get('skip', 1))
                        
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
            if endswith(model_name, '.pkl'):
                model_instance.save_model(f"{models_path_save}_{model_name}")
            else:
                model_instance.save_model(f"{models_path_save}{model_name}_{dataset_init}.pkl")

    return 

    
def models_evaluator(models : dict, test : list[np.ndarray], dataset_init: str):
    """
    Evaluates data models for a given dataset.

    Params:
    - models : dictionary of fitted models to evaluate
    - test : tuple, test set (X_test, y_test)
    - dataset_init : str, initials of the dataset to load
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


def auto_wf_baseline_each(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, tcn : bool = True, prop : float = (0.7, 0.1, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int = 42):
    """
    Return automatic workflow for 100 TCN baseline model trained each on a single time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    # Load data_config
    with open(data_config_path, 'r') as f:
        config = json.load(f)

    # Load data
    dataset = pd.read_csv(data_path)
    # save in a list all the unique time series ids
    unique_id = dataset[config[dataset_init]['id_col']].unique().tolist()

    whole_results = {}

    # for each unique time series id, create a dataset and train a TCN model
    for uid in unique_id:
        # change data_config_path to only load the time series with the current uid
        json_handler(file_path = data_config_path, id_target = uid, dataset = datasets[dataset_init])
        
        # load data
        train, val, test, X, data = dataset_handler.data_loader(dataset_init =dataset_init, data_path = data_path, data_config_path = data_config_path, is_arima = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        
        # define model
        if tcn:
            models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber'], model_type = ['TCN'], weight_type = ['uni'], data_config_path = data_config_path, model_config_paths = [model_config_path])
        else:
            models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber'], model_type = ['XGBoost'], weight_type = ['uni'], data_config_path = data_config_path, model_config_paths = [model_config_path])
        
        # train model
        if tcn:
            models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/tcn/{uid}.pkl')
        else:
            models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/xgb/{uid}.pkl')
        
        # evaluate model
        results = models_evaluator(models, test, dataset_init)

        whole_results[uid] = results

    return whole_results
    

def auto_wf_baseline_all(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, tcn : bool = True, prop : float = (0.7, 0.1, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int = 42):
    """
    Return automatic workflow for a single TCN baseline model trained on all time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    # load data
    json_handler(file_path = data_config_path, id_target = "ALL", dataset = datasets[dataset_init])
    train, val, test, X, data = dataset_handler.data_loader(dataset_init =dataset_init, data_path = data_path, data_config_path = data_config_path, is_arima = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)

    # define model
    if tcn:
        models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber'], model_type = ['TCN'], weight_type = ['uni'], data_config_path = data_config_path, model_config_paths = [model_config_path])
    else:
        models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber'], model_type = ['XGBoost'], weight_type = ['uni'], data_config_path = data_config_path, model_config_paths = [model_config_path])

    # train model
    if tcn:
        models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/tcn/all.pkl')
    else:
        models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/xgb/all.pkl')

    # evaluate model
    results = models_evaluator(models, test, dataset_init)

    return results


def auto_wf_umheme_each(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, skip : int = 12, prop : float = (0.7, 0.1, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int = 42):
    """
    Return automatic workflow for UMHEMe model trained each on a single time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    # Load data
    dataset = pd.read_csv(data_path)
    # save in a list all the unique time series ids
    unique_id = dataset[config[dataset_init]['id_col']].unique().tolist()

    whole_results = {}

    # for each unique time series id, create a dataset and train a UMHEMe model
    for uid in unique_id:
        # change data_config_path to only load the time series with the current uid
        json_handler(file_path = data_config_path, id_target = uid, dataset = datasets[dataset_init])
        
        # load data
        train, val, test, X, data = dataset_handler.data_loader(dataset_init =dataset_init, data_path = data_path, data_config_path = data_config_path, is_arima = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)

        # define model
        models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber', 'mse'], model_type = ['UMHEMe'], weight_type = ['soft_lin', 'strong_lin', 'exp'], data_config_path = data_config_path, model_config_paths = [model_config_path], base_model = 'TCN', skip = skip)
        
        # train model
        models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/umheme/uid_{uid}_skip_{skip}.pkl')
        
        # evaluate model
        results = models_evaluator(models, test, dataset_init)

        whole_results[uid] = results

    return whole_results


def auto_wf_umheme_all(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, skip : int = 12, prop : float = (0.7, 0.1, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int = 42):
    """
    Return automatic workflow for a single UMHEMe model trained on all time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    # load data
    json_handler(file_path = data_config_path, id_target = "ALL", dataset = datasets[dataset_init])
    train, val, test, X, data = dataset_handler.data_loader(dataset_init =dataset_init, data_path = data_path, data_config_path = data_config_path, is_arima = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)

    # define model
    models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber', 'mse'], model_type = ['UMHEMe'], weight_type = ['soft_lin', 'strong_lin', 'exp'], data_config_path = data_config_path, model_config_paths = [model_config_path], base_model = 'TCN', skip = skip)
    
    # train model
    models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/umheme/all.pkl')

    # evaluate model
    results = models_evaluator(models, test, dataset_init)

    return results


def auto_wf_arima(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, prop : float = (0.7, 0.1, 0.2), random_state : int = 42):
    """
    Return automatic workflow for ARIMA model.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    # load data
    train, val, test, X, data = dataset_handler.data_loader(dataset_init =dataset_init, data_path = data_path, data_config_path = data_config_path, is_arima = True, prop = prop, shuffle_data = False, shuffle_internal = True, random_state = random_state)
    
    # define model
    models = models_definer(dataset_init = [dataset_init], loss_type = ['horizon_weighted_huber'], model_type = ['ARIMA'], weight_type = ['uni'], data_config_path = data_config_path, model_config_paths = [model_config_path])

    # train model
    models_trainer(models, train, dataset_init, save_models = True, models_path_save = f'../models/{datasets[dataset_init]}/arima/all.pkl')

    # evaluate model
    results = models_evaluator(models, test, dataset_init)

    return results


def auto_workflow(dataset_init : str, data_path : str, data_config_path : str, model_config_path : str, tcn_each : bool = False, tcn_all : bool = False, xgb_each : bool = False, xgb_all : bool = False, umheme_each : bool = False, umheme_all : bool = False. arima : bool = False, prop : float = (0.7, 0.1, 0.2), shuffle_data : bool = False, shuffle_internal : bool = True, random_state : int = 42):
    """
    Automatic workflow to instantiate the dataset train/val/test sets, define, train and evaluate models.

    Params:
    - dataset_init : str, initials of the dataset to load
    
    - data_path : str, path to the dataset csv file
    - data_config_path : str, path to the dataset configuration file
    - model_config_path : str, path to the model configuration file
    
    - tcn_each : bool, whether to run TCN baseline model trained each on a single time series
    - tcn_all : bool, whether to run a single TCN baseline model trained on all time series
    - xgb_each : bool, whether to run XGBoost baseline model trained each on a single time series
    - xgb_all : bool, whether to run a single XGBoost baseline model trained on all time series
    - umheme_each : bool, whether to run UMHEMe model trained each on a single time series
    - umheme_all : bool, whether to run a single UMHEMe model trained on all time series
    - arima : bool, whether to run ARIMA model

    - prop : tuple of float, proportions for train/val/test splits
    - shuffle_data : bool, whether to shuffle the data before splitting
    - shuffle_internal : bool, whether to shuffle the internal time series before splitting
    - random_state : int, random state for reproducibility
    
    Returns:
    -------------------------------------------------------------------------------------------
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    
    whole_results = {}
    # do different subroutines for each particular case
    if tcn_each:
        tcn_each_results = auto_wf_baseline_each(dataset_init, data_path, data_config_path, model_config_path, tcn = True, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['tcn_each'] = tcn_each_results
    if tcn_all:
        tcn_all_results = auto_wf_baseline_all(dataset_init, data_path, data_config_path, model_config_path, tcn = True, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['tcn_all'] = tcn_all_results
    if xgb_each:
        xgb_each_results = auto_wf_baseline_each(dataset_init, data_path, data_config_path, model_config_path, tcn = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['xgb_each'] = xgb_each_results
    if xgb_all:
        xgb_all_results = auto_wf_baseline_all(dataset_init, data_path, data_config_path, model_config_path, tcn = False, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['xgb_all'] = xgb_all_results
    if umheme_each:
        umheme_each_results = auto_wf_umheme_each(dataset_init, data_path, data_config_path, model_config_path, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['umheme_each'] = umheme_each_results
    if umheme_all:
        umheme_all_results = auto_wf_umheme_all(dataset_init, data_path, data_config_path, model_config_path, prop = prop, shuffle_data = shuffle_data, shuffle_internal = shuffle_internal, random_state = random_state)
        whole_results['umheme_all'] = umheme_all_results
    if arima:
        arima_results = auto_wf_arima(dataset_init, data_path, data_config_path, model_config_path, prop = prop, random_state = random_state)
        whole_results['arima'] = arima_results

    return whole_results
if __name__ == "__main__":
    pass