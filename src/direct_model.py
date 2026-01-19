import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import xgboost as xgb

import json


class DirectModel:
    """
    DirectModel is the class implementing the direct strategy for multi-step time series forecasting.
    For now, a Temporary Convolutional Neural Network is used as the underlying model.
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)

        self.window = config["window"]
        self.horizon = config["horizon"]

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


class TCN(nn.Module, DirectModel):
    def __init__(self, file_path: str):
        nn.Module.__init__(self)

        with open(file_path, 'r') as file:
            config = json.load(file)

        self.window = config['window']
        self.horizon = config['horizon']

        inner_layers_dim = config['inner_layers_dim']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']

        self.n_epochs = config['n_epochs']

        self.device = config['device']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.activation = getattr(f, config['activation'])
        self.loss = nn.MSELoss()

        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(
            in_channels=1,
            out_channels=inner_layers_dim[0],
            kernel_size=kernel_size[0],
            stride=stride,
            padding=padding
        ))
        for i in range(config['n_layers'] - 2):
            self.conv_layers.append(nn.Conv1d(
                in_channels=inner_layers_dim[i],
                out_channels=inner_layers_dim[i+1],
                kernel_size=kernel_size[i],
                stride=stride,
                padding=padding
            ))
        self.conv_layers.append(nn.Conv1d(
            in_channels=inner_layers_dim[-2],
            out_channels=inner_layers_dim[-1],
            kernel_size=kernel_size[-1],
            stride=stride,
            padding=padding
        ))
        # Features of the last timestamp to the horizon
        self.readout = nn.Linear(inner_layers_dim[-1], self.horizon)

        self.optim = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])

        self.to(self.device)


    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        x = x[:, :, -1]
        x = self.readout(x)
        return x


    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        print("Starting TCN training...")

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        print("Training TCN model...")

        self.train()

        epoch_iterator = tqdm(range(self.n_epochs), desc="Training TCN")

        for epoch in epoch_iterator:
            epoch_loss = 0.0

            print(f"Epoch {epoch+1}/{self.n_epochs}")

            for batch_X, batch_y in loader:

                print("Processing new batch...")

                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward Pass
                self.optim.zero_grad()
                y_hat = self(batch_X)

                # Compute Loss
                loss = self.loss(y_hat, batch_y)

                # Backward Pass
                loss.backward()
                self.optim.step()

                # Accumulate loss (multiply by batch size to get total error, then avg later)
                epoch_loss += loss.item() * batch_X.size(0)

            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(dataset)

            # Update progress bar every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #    epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        return avg_loss


    def predict(self, X: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.eval()
        with torch.no_grad():
            y_hat = self(X_tensor)
        return y_hat.cpu().numpy() # Move back to CPU

    def save_model(self, file_path: str):
        """
        Save the model's state dictionary to the specified file path.

        Params:
        - file_path : str

        Returns:
        - None
        """
        torch.save(self.state_dict(), file_path)
        return


class XGBoost(DirectModel):
    def __init__(self, file_path: str):
        super().__init__(file_path)

        with open(file_path, 'r') as file:
            config = json.load(file)

        self.window = config['window']
        self.horizon = config['horizon']

        self.reg = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            tree_method="hist",
            device=config["device"] if config["device"] == 'cuda' else None
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
        Save the model to the specified file path.

        Params:
        - file_path : str

        Returns:
        - None
        """
        self.reg.save_model(file_path)
        return


if __name__ == '__main__':
    pass