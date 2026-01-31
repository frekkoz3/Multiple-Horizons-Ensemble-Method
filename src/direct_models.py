import numpy as np

import json
import joblib
import os

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults

import pmdarima as pm # used for auto_arima 


from src.metrics import *


class DirectModel:
    """
    DirectModel is the class implementing the direct strategy for multi-step time series forecasting.
    For now, a Temporary Convolutional Neural Network and a XGBoost model are used as the underlying models.
    """

    def __init__(self, horizon : int, config_path: str):

        self.horizon = horizon # this is unique for each model so it should be passed

        with open(config_path, 'r') as file:
            config = json.load(file)

        self.window = config["window"] # this is shared across all the models so it is correct to take it from the config

    def fit(self, X : np.ndarray, y : np.ndarray):
        """
            Fit based on the error committed by predicting y_hat using X as input.

            Params:
            - X : input of size self.window
            - y : ground truth of size self.horizon

            Returns : 
            - eps : errors committed during the prediction
        """
        pass


    def predict(self, X):
        """
            Predict y_hat using X as input.

            Params : 
            - X : input of size self.window 

            Return:
            - y_hat : output of size self.horizon
        """
        pass

    def __str__(self):
        return f"class : {self.__class__}; horizon : {self.horizon}"


class TCN(nn.Module, DirectModel):
    def __init__(self, file_path: str, horizon : int | None = None, window : int | None = None):
        nn.Module.__init__(self)

        with open(file_path, 'r') as file:
            config = json.load(file)


        # ---- Shared Hyperparameters ----
        self.window = config['window'] if window is None else window
        self.horizon = config['horizon'] if horizon is None else horizon


        # ---- TCN Hyperparameters ----
        self.inner_layers_dim = config['inner_layers_dim']
        self.kernel_size = config['kernel_size']
        self.stride = config['stride']
        self.padding = config['padding']

        self.n_epochs = config['n_epochs']

        self.device = config['device']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.activation = getattr(f, config['activation'])

        if config['loss'] == "horizon_weighted_huber":
            self.loss = horizon_weighted_huber
        else:
            self.loss = mse


        # ---- Weights for loss ----
        self.w_decay = config['weights_decay']
        self.set_weights()


        # ---- Build TCN layers ----
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(
            in_channels=1,
            out_channels=self.inner_layers_dim[0],
            kernel_size=self.kernel_size[0],
            stride=self.stride,
            padding=self.padding
        ))
        for i in range(config['n_layers'] - 2):
            self.conv_layers.append(nn.Conv1d(
                in_channels=self.inner_layers_dim[i],
                out_channels=self.inner_layers_dim[i+1],
                kernel_size=self.kernel_size[i],
                stride=self.stride,
                padding=self.padding
            ))
        self.conv_layers.append(nn.Conv1d(
            in_channels=self.inner_layers_dim[-2],
            out_channels=self.inner_layers_dim[-1],
            kernel_size=self.kernel_size[-1],
            stride=self.stride,
            padding=self.padding
        ))
        self.readout = nn.Linear(self.inner_layers_dim[-1], self.horizon)       # Features of the last timestamp to the horizon

        self.optim = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])

        # ---- Move to device ----
        self.to(self.device)


    def set_weights(self):
        if self.w_decay == "uni":
            self.weights = torch.ones(self.horizon)
        elif self.w_decay == "soft_lin":
            self.weights = torch.linspace(1.0, 5, self.horizon)
        elif self.w_decay == "strong_lin":
            self.weights = torch.arange(1, self.horizon + 1)
        elif self.w_decay == "exp":
            gamma = 1.3  # >1 emphasizes long horizon
            self.weights = gamma ** torch.arange(self.horizon)


    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        x = x[:, :, -1]
        x = self.readout(x)
        return x


    def fit(self, X: np.ndarray, y: np.ndarray, verbose : bool = False):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        if verbose:
            print("Starting TCN training...")

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if verbose:
            print("Training TCN model...")

        self.train()

        epoch_iterator = tqdm(range(self.n_epochs), desc="Training TCN")

        for epoch in epoch_iterator:
            epoch_loss = 0.0
            if verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}")

            for batch_X, batch_y in loader:
                if verbose:
                    print("Processing new batch...")

                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward Pass
                self.optim.zero_grad()
                y_hat = self(batch_X)

                # Compute Loss
                loss = self.loss(y_hat, batch_y, self.weights)

                # Backward Pass
                loss.backward()
                self.optim.step()

                # Accumulate loss (multiply by batch size to get total error, then avg later)
                epoch_loss += loss.item() * batch_X.size(0)

            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(dataset)

            if verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        return avg_loss


    def predict(self, X: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.eval()
        with torch.no_grad():
            y_hat = self(X_tensor)
        
        return y_hat.cpu().numpy() # Move back to CP

        
    def save_model(self, file_path: str):
        """
        Save the model's state dictionary to the specified file path.

        Params:
        - file_path : str

        Returns:
        - None
        """
        if file_path.endswith(".pkl"):
            torch.save(self.state_dict(), file_path)
        else:
            torch.save(self.state_dict(), f"{file_path}.pkl")
        return

    @classmethod
    def load_model(cls, file_path: str, config_path: str = None):
        """
        Smart loader for TCN.
        - file_path: Path to model file (.pt or legacy .pkl)
        - config_path: Path to .json config (REQUIRED for legacy .pkl files)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")

        # Attempt to load with torch
        try:
            checkpoint = torch.load(file_path, map_location='cpu') # Safer for CPU inference
        except Exception as e:
            raise ValueError(f"Failed to load torch model: {e}")

        if isinstance(checkpoint, dict):
            if config_path is None:
                raise ValueError(f"Legacy model at {file_path} requires 'config_path' to initialize architecture.")
            
            # Init architecture from external JSON
            model = cls(file_path=config_path)
            model.load_state_dict(checkpoint)
            print(f"Loaded legacy weights from {file_path} using config {config_path}")
            
        else:
            raise ValueError("Unknown TCN file format.")

        model.eval()
        return model
        


class XGBoost(DirectModel):
    def __init__(self, file_path: str, horizon : int | None = None, window : int | None = None):
        super().__init__(horizon, file_path)

        with open(file_path, 'r') as file:
            config = json.load(file)

        self.window = config['window'] if window is None else window
        self.horizon = config['horizon'] if horizon is None else horizon

        if config['loss'] == "horizon_weighted_huber":
            self.objective = horizon_weighted_huber
        else:
            self.objective = "reg:squarederror"

        self.reg = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            tree_method="hist",
            device=config["device"] if config["device"] == 'cuda' else None,
            objective=self.objective
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.reg.fit(X, y, eval_set=[(X, y)], verbose=False)
        # Access the history
        results = self.reg.evals_result()

        # 'validation_0' is the name assigned to the first item in eval_set
        # 'rmse' is the default metric for reg:squarederror (Root MSE)
        # XGBoost typically tracks RMSE, so we square it to get MSE
        final_rmse = results['validation_0']['rmse'][-1]

        return final_rmse ** 2

    
    def predict(self, X: np.ndarray):
        return self.reg.predict(X)

    
    def save_model(self, file_path: str):
        """
        Saves the entire class instance (Config + Weights + Wrapper).
        """
        # Ensure we use .pkl extension for Python objects
        if file_path.endswith('.json'):
            file_path = file_path.replace('.json', '.pkl')
        elif not file_path.endswith('.pkl'):
            file_path += '.pkl'

        # joblib.dump saves 'self', which includes:
        joblib.dump(self, file_path)
        print(f"XGBoost model (full object) saved to {file_path}")

    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads the entire class instance.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # This restores the object exactly as it was before saving
        model = joblib.load(file_path)
        return model


class ARIMAModel(DirectModel):
    """
    ARIMA and SARIMA models. They are notì "direct" but autoregressive models, and thus used only as benchmarks. 
    """
    def __init__(self, file_path : str, horizon : int | None = None):
        super().__init__(horizon, file_path)
        
        with open(file_path, 'r') as f:
            config = json.load(f)

        self.horizon = config['horizon'] if horizon is None else horizon
        self.window = config['window']

        # --- Parameters ----
        self.auto_arima = config['auto_arima']
        self.seasonality = config['seasonality']
        
        if not self.auto_arima:
            self.p = config['p']
            self.d = config['d']
            self.q = config['q']
            if self.seasonality > 1:
                self.P = config['P']
                self.D = config['D']
                self.Q = config['Q']

        self.skip = config['skip']

    def fit(self, y: np.ndarray):
        if self.auto_arima:
            self.model = pm.auto_arima(y=y, m=self.seasonality)
        else:
            # Statsmodels definition
            if self.seasonality == 1:
                model_def = ARIMA(endog=y, order=(self.p, self.d, self.q))
            else:
                model_def = ARIMA(endog=y, order=(self.p, self.d, self.q), 
                                  seasonal_order=(self.P, self.D, self.Q, self.seasonality))
            self.model = model_def.fit()
            
            return self.model.summary()


    def predict(self, test: np.ndarray):
        y_hat = []
        errors = [] 

        #at the beginning, we set in memory the model fitted on the training set + the first self.window values of the test set
        memory = list(test[:self.window])
        if self.auto_arima:
            self.model.update(memory)
        else:
            self.model = self.model.extend(memory)
        

        for t in range(self.window, len(test), self.skip):
            # forecast
            if self.auto_arima:
                # Pmdarima uses predict
                n_periods = self.horizon if t+self.horizon < len(test) else len(test)-t
                forecast = self.model.predict(n_periods=n_periods) 
            else:
                # Statsmodels uses forecast
                forecast = self.model.forecast(steps=self.horizon)
            
            y_hat.append(forecast)

            forecast = forecast[:len(test[t:t + self.horizon])] # we are at the end of the time series test set !
            
            errors.append(test[t:t + self.horizon] - forecast) 

            # update ARIMA with REAL value ---
            true_values = list(test[t:t + self.skip])
            if self.auto_arima:
                self.model.update(true_values)
            else:
                self.model = self.model.extend(true_values)

        mse_error = np.mean([np.mean(err**2) for err in errors])

        return mse_error
        

    def save_model(self, file_path: str):
        """
        Saves the ENTIRE wrapper class (Configuration + Internal Model).
        This works for both pmdarima and statsmodels backends.
        """
        # Ensure correct extension
        if file_path.endswith('.json'):
             file_path = file_path.replace('.json', '.pkl')
        elif not file_path.endswith('.pkl'):
             file_path += '.pkl'

        # Check if fitted
        if self.model is None:
            print(f"Warning: Saving an unfitted ARIMA model to {file_path}")

        # joblib handles the internal serialization of pmdarima or statsmodels objects automatically
        joblib.dump(self, file_path)
        print(f"ARIMA model (full wrapper) saved to {file_path}")

    
    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads the ENTIRE wrapper class.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model file found at {file_path}")
        
        # This restores the wrapper, the config, AND the correct inner model type
        model = joblib.load(file_path)
        return model


if __name__ == '__main__':
    pass