import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from src.direct_models import *
from src.mheme import *
from src.config_files import *
from src.data_handler import *

from statsmodels.tsa.statespace.tools import diff
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def models_definer(dataset_init : list[str],
                   loss_type : list[str],
                   model_type : list[str],
                   weight_type : list[str],
                   data_config_path : str,
                   model_config_paths : list[str], 
                   **kwargs) -> dict:
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
                    if model == "ARIMA":  
                        json_handler(file_path = model_config_paths[j], 
                                    weights_decay = weights, 
                                    loss_type = loss, 
                                    horizon = data_config[models_names[ds]]['horizon'],
                                    window = data_config[models_names[ds]]['window'],
                                    skip = data_config[models_names[ds]]['skip']
                                    )
                    else:
                        json_handler(file_path = model_config_paths[j], 
                                    weights_decay = weights, 
                                    loss_type = loss, 
                                    horizon = data_config[models_names[ds]]['horizon'],
                                    window = data_config[models_names[ds]]['window']
                                    )
                    
                    # Define model
                    if model == 'TCN':
                        models[f'TCN_{loss}_{weights}'] = TCN(file_path = model_config_paths[j])
                    elif model == 'XGBoost':
                        models[f'XGBoost_{loss}_{weights}'] = XGBoost(file_path = model_config_paths[j])
                    elif model == 'ARIMA':
                        models[f'ARIMA_{loss}_{weights}'] = ARIMAModel(file_path = model_config_paths[j])
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

                            models[f'UMHEMe_{loss}_{weights}'] = UMHEMe(horizon = dataset_horizon, 
                                                                        window = dataset_window, 
                                                                        model_class = base_model_class, 
                                                                        config_path = model_config_paths[j], 
                                                                        skip = kwargs.get('skip', 1))
                        
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
        
        if not "ARIMA" in model_name:
            model_instance.fit(train[0], train[1])
        else:
            model_instance.fit(train[0])

        # if model is UMHEME, compute weights
        if "UMHEMe" in model_name:
            model_instance.compute_weights(train[0], train[1])

        if save_models:
            os.makedirs(models_path_save, exist_ok=True)
            if np.strings.endswith(model_name, '.pkl'):
                model_instance.save_model(f"{models_path_save}/{model_name}")
            else:
                model_instance.save_model(f"{models_path_save}/{model_name}_{dataset_init}.pkl")

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
    
    # Loop through all models regardless of count
    for model_name, model_instance in models.items():
        print(f"\nEvaluate model: {model_name}")
        
        results[model_name] = model_instance.predict(test[0])
    
    if len(results) == 1:
        single_prediction = next(iter(results.values()))
        return single_prediction
    
    return results


def models_loader(file_path: str, config_path : str | None = None, is_arima = False):
    """
    Loads ANY model (TCN, XGB, ARIMA, UMHEMe).
    Automatically detects if it's a legacy TCN weight dict or a new Full Object.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    if is_arima:
        try:
            return ARIMAModel.load_model(config_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    else:

        # Try loading as a standard Pickle/Joblib object first
        try:
            obj = joblib.load(file_path)
            
            # Check if it's a raw dictionary (Legacy TCN weights)
            if isinstance(obj, dict) and 'config' not in obj:
                raise ValueError("Detected Legacy TCN Weights (Dict)")
            
            return obj # It's a full object (XGB, ARIMA, UMHEMe)

        except (ValueError, Exception):
            # If joblib fails or we found a dict, try Torch Loader (for TCN)
            try:
                # This calls our smart TCN loader which handles both New and Old formats
                # We pass TCN_PATH_CONFIG_LOAD as a fallback for legacy files
                return TCN.load_model(file_path, config_path=config_path)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                return None


def models_saver(dataset_init : str, errors: dict):
    # ---------------------------------------------------------
    # SAVING SUBROUTINE
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. FLATTEN DATA FOR CSV (Analysis Ready)
    # This converts the nested dictionary into a list of records
    records = []
    for method_name, errors_dict in errors.items():
        # errors_dict is e.g. {'ts_01': 0.05} or {'whole_dataset': 0.05}
        for uid, error_val in errors_dict.items():
            records.append({
                'Method': method_name,
                'ID': uid,
                'Error': float(error_val) # Ensure numpy floats are converted
            })
    
    if records:
        df_results = pd.DataFrame(records)
        csv_filename = f"../models/results/errors_{dataset_init}_{timestamp}.csv"
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df_results.to_csv(csv_filename, index=False)
        print(f"Results (CSV) saved to: {csv_filename}")
    
    # 2. SAVE RAW JSON (Backup)
    # Necessary for full reproducibility or if data becomes non-tabular later
    class TypeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            return super().default(obj)

    json_filename = f"../models/results/errors_{dataset_init}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(errors, f, indent=4, cls=TypeEncoder)
    print(f"Results (JSON) saved to: {json_filename}")


def auto_wf_baseline_each(dataset_init : str, 
                          data_path : str, 
                          data_config_path : str, 
                          model_config_paths : str | list[str], 
                          tcn : bool = True, 
                          prop : float = (0.7, 0.1, 0.2), 
                          shuffle_data : bool = False, 
                          shuffle_internal : bool = True, 
                          random_state : int = 42, 
                          pretrained_model : bool = False):
    """
    Return automatic workflow for 100 TCN baseline model trained each on a single time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    
    # Load data_config
    with open(data_config_path, 'r') as f:
        config = json.load(f)

    # Load data
    dataset = pd.read_csv(data_path+f'/{datasets[dataset_init]}.csv')
    # save in a list all the unique time series ids
    unique_id = dataset[config[datasets[dataset_init]]['id_col']].unique().tolist()

    whole_predictions = {}
    whole_errors = {}

    # for each unique time series id, create a dataset and train a TCN model
    for uid in unique_id:
        print(f'\n--------------------\nProcessing time series with ID:\t{uid}\n--------------------')
        # change data_config_path to only load the time series with the current uid
        models_path_save = f'../models/{datasets[dataset_init]}/tcn/{uid}' if tcn else f'../models/{datasets[dataset_init]}/xgb/{uid}'

        json_handler(file_path = data_config_path, id_target = uid, dataset = datasets[dataset_init])
        
        # load data
        train, val, test, X, data = dataset_handler(dataset_init =dataset_init,
                                                    data_path = data_path,
                                                    data_config_path = data_config_path,
                                                    is_arima = False,
                                                    prop = prop,
                                                    shuffle_data = shuffle_data,
                                                    shuffle_internal = shuffle_internal,
                                                    random_state = random_state
                                                    )
        
        # define model
        if tcn:
            model_config_path = model_config_paths[0] if isinstance(model_config_paths, list) else model_config_paths
            models = models_definer(dataset_init = [dataset_init],
                                    loss_type = ['horizon_weighted_huber'],
                                    model_type = ['TCN'],
                                    weight_type = ['uni'],
                                    data_config_path = data_config_path,
                                    model_config_paths = [model_config_path]
                                    )
        else:
            model_config_path = model_config_paths[1] if isinstance(model_config_paths, list) else model_config_paths
            models = models_definer(dataset_init = [dataset_init],
                                    loss_type = ['mse'],
                                    model_type = ['XGBoost'],
                                    weight_type = ['uni'],
                                    data_config_path = data_config_path,
                                    model_config_paths = [model_config_path]
                                    )

        if not pretrained_model: 
            # train model
            if tcn:
                models_trainer(models, train, dataset_init, save_models = True, models_path_save = models_path_save)
            else:
                models_trainer(models, train, dataset_init, save_models = True, models_path_save = models_path_save)
        else:
            # models are already pre-trained
            model_name, _ = list(models.items())[0]
            trained_model_path = models_path_save + f'/{model_name}_{dataset_init}.pkl'
            trained_model = models_loader(file_path = trained_model_path, 
                                     config_path = model_config_path
                                    )
            models[model_name] = trained_model
            
        # evaluate model
        prediction = models_evaluator(models, test, dataset_init)
        error = mse(prediction, test[1])

        whole_predictions[uid] = prediction
        whole_errors[uid] = error

    return whole_errors
    

def auto_wf_baseline_all(dataset_init : str,
                         data_path : str,
                         data_config_path : str,
                         model_config_paths : str | list[str],
                         tcn : bool = True,
                         prop : float = (0.7, 0.1, 0.2),
                         shuffle_data : bool = False,
                         shuffle_internal : bool = True,
                         random_state : int = 42,
                         pretrained_model : bool = False
                         ):
    """
    Return automatic workflow for a single TCN baseline model trained on all time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    models_path_save = f'../models/{datasets[dataset_init]}/tcn/all' if tcn else f'../models/{datasets[dataset_init]}/xgb/all'
    
    # load data
    json_handler(file_path = data_config_path, id_target = "ALL", dataset = datasets[dataset_init])
    train, val, test, X, data = dataset_handler(dataset_init =dataset_init,
                                                data_path = data_path,
                                                data_config_path = data_config_path,
                                                is_arima = False,
                                                prop = prop,
                                                shuffle_data = shuffle_data,
                                                shuffle_internal = shuffle_internal,
                                                random_state = random_state
                                                )

    # define model
    if tcn:
        model_config_path = model_config_paths[0] if isinstance(model_config_paths, list) else model_config_paths
        models = models_definer(dataset_init = [dataset_init],
                                loss_type = ['horizon_weighted_huber'],
                                model_type = ['TCN'],
                                weight_type = ['uni'],
                                data_config_path = data_config_path,
                                model_config_paths = [model_config_path]
                                )
    else:
        model_config_path = model_config_paths[1] if isinstance(model_config_paths, list) else model_config_paths
        models = models_definer(dataset_init = [dataset_init],
                                loss_type = ['mse'],
                                model_type = ['XGBoost'],
                                weight_type = ['uni'],
                                data_config_path = data_config_path,
                                model_config_paths = [model_config_path]
                                )

    # train model
    if not pretrained_model:
        if tcn:
            models_trainer(models, train, dataset_init, save_models = True, models_path_save = models_path_save)
        else:
            models_trainer(models, train, dataset_init, save_models = True, models_path_save = models_path_save)
    else:
        # models are already pre-trained
        model_name, _ = list(models.items())[0]
        trained_model_path = models_path_save + f'/{model_name}_{dataset_init}.pkl'
        trained_model = models_loader(file_path = trained_model_path, 
                                 config_path = model_config_path
                                )
        models[model_name] = trained_model
        
    # evaluate model
    prediction = models_evaluator(models, test, dataset_init)
    error = mse(prediction, test[1])
    error_dict = {'whole_dataset' : error}

    return error_dict


def auto_wf_umheme_each(dataset_init : str,
                        data_path : str,
                        data_config_path : str,
                        model_config_paths : str | list[str],
                        prop : float = (0.7, 0.1, 0.2),
                        shuffle_data : bool = False,
                        shuffle_internal : bool = True,
                        random_state : int = 42,
                        skip : int = 1,
                        weight_type : list[str] | None = ['soft_lin', 'strong_lin', 'exp'],
                        loss_type : list[str] | None = ['horizon_weighted_huber', 'mse'],
                        pretrained_model : bool = False
                        ):
    """
    Return automatic workflow for UMHEMe model trained each on a single time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    
    # Load data_config
    with open(data_config_path, 'r') as f:
        config = json.load(f)
    
    # Load data
    dataset = pd.read_csv(data_path+f'/{datasets[dataset_init]}.csv')
    # save in a list all the unique time series ids
    unique_id = dataset[config[datasets[dataset_init]]['id_col']].unique().tolist()

    whole_predictions = {}
    whole_errors = {}

    # for each unique time series id, create a dataset and train a UMHEMe model
    for uid in unique_id:
        print(f'\n--------------------\nProcessing time series with ID:\t{uid}\n--------------------')
        # change data_config_path to only load the time series with the current uid
        json_handler(file_path = data_config_path, id_target = uid, dataset = datasets[dataset_init])
        models_path_save = f'../models/{datasets[dataset_init]}/umheme/uid_{uid}_skip_{skip}'

        # load data
        train, val, test, X, data = dataset_handler(dataset_init =dataset_init,
                                                    data_path = data_path,
                                                    data_config_path = data_config_path,
                                                    is_arima = False,
                                                    prop = prop,
                                                    shuffle_data = shuffle_data,
                                                    shuffle_internal = shuffle_internal,
                                                    random_state = random_state
                                                    )

        # define model
        model_config_path = model_config_paths[0] if isinstance(model_config_paths, list) else model_config_paths
        models = models_definer(dataset_init = [dataset_init],
                                loss_type = loss_type,
                                model_type = ['UMHEMe'],
                                weight_type = weight_type,
                                data_config_path = data_config_path,
                                model_config_paths = [model_config_path],
                                base_model = 'TCN',
                                skip = skip
                                )
        
        # train model
        if not pretrained_model:
                models_trainer(models,
                               train,
                               dataset_init,
                               save_models = True,
                               models_path_save = models_path_save
                               )
        else:
            # models are already pre-trained
            model_name, _ = list(models.items())[0]
            trained_model_path = models_path_save + f'/{model_name}_{dataset_init}.pkl'
            trained_model = models_loader(file_path = trained_model_path, 
                                     config_path = model_config_path
                                    )
            models[model_name] = trained_model
            
        
        # evaluate model
        prediction = models_evaluator(models, test, dataset_init)
        error = mse(prediction, test[1])
        
        whole_predictions[uid] = prediction
        whole_errors[uid] = error
        
    return whole_errors


def auto_wf_umheme_all(dataset_init : str,
                       data_path : str,
                       data_config_path : str,
                       model_config_paths : str | list[str],
                       prop : float = (0.7, 0.1, 0.2),
                       shuffle_data : bool = False,
                       shuffle_internal : bool = True,
                       random_state : int = 42,
                       skip : int = 1,
                       weight_type : list[str] | None = ['soft_lin', 'strong_lin', 'exp'],
                       loss_type : list[str] | None = ['horizon_weighted_huber', 'mse'],
                       pretrained_model : bool = False
                       ):
    """
    Return automatic workflow for a single UMHEMe model trained on all time series.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    models_path_save = f'../models/{datasets[dataset_init]}/umheme/all'

    # load data
    json_handler(file_path = data_config_path, id_target = "ALL", dataset = datasets[dataset_init])

    train, val, test, X, data = dataset_handler(dataset_init =dataset_init,
                                                data_path = data_path,
                                                data_config_path = data_config_path,
                                                is_arima = False,
                                                prop = prop,
                                                shuffle_data = shuffle_data,
                                                shuffle_internal = shuffle_internal,
                                                random_state = random_state
                                                )
    # define model
    model_config_path = model_config_paths[0] if isinstance(model_config_paths, list) else model_config_paths
    models = models_definer(dataset_init = [dataset_init],
                            loss_type = loss_type,
                            model_type = ['UMHEMe'],
                            weight_type = weight_type,
                            data_config_path = data_config_path, 
                            model_config_paths = [model_config_path],
                            base_model = 'TCN',
                            skip = skip)
    
    # train model
    if not pretrained_model:
        models_trainer(models,
                       train,
                       dataset_init,
                       save_models = True,
                       models_path_save = models_path_save
                       )
    else:
        # models are already pre-trained
        model_name, _ = list(models.items())[0]
        trained_model_path = models_path_save + f'/{model_name}_{dataset_init}.pkl'
        trained_model = models_loader(file_path = trained_model_path, 
                                 config_path = model_config_path
                                )
        models[model_name] = trained_model

    # evaluate model
    prediction = models_evaluator(models, test, dataset_init)
    error = mse(prediction, test[1])
    error_dict = {'whole_dataset': error}

    return error_dict


def auto_wf_arima_each(dataset_init : str,
                       data_path : str,
                       data_config_path : str,
                       model_config_paths : str | list[str],
                       prop : float = (0.7, 0.1, 0.2),
                       random_state : int = 42,
                       shuffle_data : bool = False,
                       shuffle_internal : bool = False,
                       pretrained_model : bool = False
                       ):
    """
    Return automatic workflow for ARIMA model.
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"
    
    
    # Load data_config
    with open(data_config_path, 'r') as f:
        config = json.load(f)
    
    # Load data
    dataset = pd.read_csv(data_path+f'/{datasets[dataset_init]}.csv')
    # save in a list all the unique time series ids
    unique_id = dataset[config[datasets[dataset_init]]['id_col']].unique().tolist()

    whole_errors = {}

    # for each unique time series id, create a dataset and train a UMHEMe model
    for uid in unique_id:
        print(f'\n--------------------\nProcessing time series with ID:\t{uid}\n--------------------')
        # change data_config_path to only load the time series with the current uid
        json_handler(file_path = data_config_path, id_target = uid, dataset = datasets[dataset_init])
        models_path_save = f'../models/{datasets[dataset_init]}/arima/{uid}'
        
        # load data
        train, val, test, X, data = dataset_handler(dataset_init =dataset_init,
                                                    data_path = data_path,
                                                    data_config_path = data_config_path,
                                                    is_arima = True,
                                                    prop = prop,
                                                    shuffle_data = shuffle_data,
                                                    shuffle_internal = shuffle_internal,
                                                    random_state = random_state)
        # define model
        model_config_path = model_config_paths[2] if isinstance(model_config_paths, list) else model_config_paths
        with open(data_config_path, 'r') as f:
            config = json.load(f)

        skip = config[datasets[dataset_init]]['skip']

        json_handler(file_path = model_config_path, skip = skip)

        models = models_definer(dataset_init = [dataset_init],
                                loss_type = ['mse'],
                                model_type = ['ARIMA'],
                                weight_type = ['uni'],
                                data_config_path = data_config_path,
                                model_config_paths = [model_config_path]
                                )

        if not pretrained_model:
            # train model
            models_trainer(models, train, dataset_init, save_models = True, models_path_save = models_path_save)
        else:
            # models are already pre-trained
            model_name, _ = list(models.items())[0]
            trained_model_path = models_path_save + f'/{model_name}_{dataset_init}.pkl'
            trained_model = models_loader(file_path = trained_model_path, 
                                     config_path = model_config_path,
                                     is_arima=True
                                    )
            models[model_name] = trained_model.rebuild_model(train[0]) # should work
            
        # evaluate model
        error = models_evaluator(models, test, dataset_init)
        whole_errors[uid] = error

    return whole_errors


def auto_workflow(dataset_init : str,
                  data_path : str,
                  data_config_path : str,
                  model_config_paths : str | list[str],
                  tcn_each : bool = False,
                  tcn_all : bool = False,
                  xgb_each : bool = False,
                  xgb_all : bool = False,
                  umheme_each : bool = False,
                  umheme_all : bool = False,
                  arima : bool = False,
                  prop : float = (0.7, 0.1, 0.2),
                  shuffle_data : bool = False,
                  shuffle_internal : bool = True,
                  random_state : int = 42,
                  skip_umheme : int = 1,
                  weight_type : list[str] | None = ['soft_lin', 'strong_lin', 'exp'],
                  loss_type : list[str] | None = ['horizon_weighted_huber', 'mse'],
                  pretrained_model : bool = False
                  ):
    """
    Automatic workflow to instantiate the dataset train/val/test sets, define, train and evaluate models.

    Params:
    - dataset_init : str, initials of the dataset to load
    
    - data_path : str, path to the dataset csv file
    - data_config_path : str, path to the dataset configuration file
    - model_config_paths : str or list of str, path(s) to the model configuration file
    
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

    - pretrained_model : bool, whether to load or not pretrained models
    
    Returns:
    -------------------------------------------------------------------------------------------
    """
    datasets = {'e': 'electricity', 's': 'solar', 't': 'traffic', 'v': 'volatility', 'w': 'wind'}
    assert dataset_init in datasets, f"Dataset initials {dataset_init} not recognized. Choose among 'e', 's', 't', 'v', 'w'"

    
    whole_errors = {}
    # do different subroutines for each particular case
    if tcn_each:
        tcn_each_errors = auto_wf_baseline_each(dataset_init,
                                                data_path,
                                                data_config_path,
                                                model_config_paths,
                                                tcn = True,
                                                prop = prop,
                                                shuffle_data = shuffle_data,
                                                shuffle_internal = shuffle_internal,
                                                random_state = random_state,
                                                pretrained_model = pretrained_model
                                                )
        whole_errors['tcn_each'] = tcn_each_errors
    if tcn_all:
        tcn_all_errors = auto_wf_baseline_all(dataset_init,
                                              data_path,
                                              data_config_path,
                                              model_config_paths,
                                              tcn = True,
                                              prop = prop,
                                              shuffle_data = shuffle_data,
                                              shuffle_internal = shuffle_internal,
                                              random_state = random_state,
                                              pretrained_model = pretrained_model
                                              )
        whole_errors['tcn_all'] = tcn_all_errors
    if xgb_each:
        xgb_each_errors = auto_wf_baseline_each(dataset_init,
                                                data_path,
                                                data_config_path,
                                                model_config_paths,
                                                tcn = False,
                                                prop = prop,
                                                shuffle_data = shuffle_data,
                                                shuffle_internal = shuffle_internal,
                                                random_state = random_state,
                                                pretrained_model = pretrained_model
                                                )
        whole_errors['xgb_each'] = xgb_each_errors
    if xgb_all:
        xgb_all_errors = auto_wf_baseline_all(dataset_init,
                                              data_path,
                                              data_config_path,
                                              model_config_paths,
                                              tcn = False, 
                                              prop = prop,
                                              shuffle_data = shuffle_data,
                                              shuffle_internal = shuffle_internal,
                                              random_state = random_state,
                                              pretrained_model = pretrained_model
                                              )
        whole_errors['xgb_all'] = xgb_all_errors
    if umheme_each:
        umheme_each_errors = auto_wf_umheme_each(dataset_init,
                                                 data_path,
                                                 data_config_path,
                                                 model_config_paths,
                                                 prop = prop,
                                                 shuffle_data = shuffle_data,
                                                 shuffle_internal = shuffle_internal,
                                                 random_state = random_state,
                                                 skip = skip_umheme,
                                                 weight_type = weight_type,
                                                 loss_type = loss_type,
                                                 pretrained_model = pretrained_model
                                                 )
        whole_errors['umheme_each'] = umheme_each_errors
    if umheme_all:
        umheme_all_errors = auto_wf_umheme_all(dataset_init,
                                               data_path,
                                               data_config_path,
                                               model_config_paths,
                                               prop = prop,
                                               shuffle_data = shuffle_data,
                                               shuffle_internal = shuffle_internal,
                                               random_state = random_state,
                                               skip = skip_umheme,
                                               weight_type = weight_type,
                                               loss_type = loss_type,
                                               pretrained_model = pretrained_model
                                               )
        whole_errors['umheme_all'] = umheme_all_errors
    if arima:
        arima_errors = auto_wf_arima_each(dataset_init,
                                          data_path,
                                          data_config_path,
                                          model_config_paths,
                                          prop = prop,
                                          random_state = random_state,
                                          pretrained_model = pretrained_model
                                          )
        whole_errors['arima'] = arima_errors

    # save whole_errors as json
    models_saver(dataset_init = dataset_init, errors = whole_errors)
    
    return whole_errors
    
if __name__ == "__main__" :
    
    pass