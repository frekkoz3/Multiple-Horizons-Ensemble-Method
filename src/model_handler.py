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


if __name__ == "__main__":
    pass