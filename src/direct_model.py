import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import xgboost as xgb


class DirectModel:
    """
    DirectModel is the class implementing the direct strategy for multi-step time series forecasting.
    For now, a Temporary Convolutional Neural Network is used as the underlying model.
    """

    def __init__(self, window : int, horizon : int):
        self.window = window
        self.horizon = horizon

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
    def __init__(self, window: int, horizon: int, n_layers: int, inner_layers_dim: list[int],
                 kernel_size: list[int], stride: int, padding: int, activation: str,
                 learning_rate: float, n_epochs: int, batch_size : int, device: str):

        nn.Module.__init__(self)
        DirectModel.__init__(self, window, horizon)

        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.activation = getattr(f, activation)
        self.loss = nn.MSELoss()

        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(
            in_channels=1,
            out_channels=inner_layers_dim[0],
            kernel_size=kernel_size[0],
            stride=stride,
            padding=padding
        ))
        for i in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(
                in_channels=inner_layers_dim[i],
                out_channels=inner_layers_dim[i+1],
                kernel_size=kernel_size[i+1],
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
        self.readout = nn.Linear(inner_layers_dim[-1], horizon)

        # Initialize Optimizer
        self.to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        x = x[:, :, -1]
        x = self.readout(x)
        return x

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.train()

        epoch_iterator = tqdm(range(self.n_epochs), desc="Training TCN")

        for epoch in epoch_iterator:
            epoch_loss = 0.0

            for batch_X, batch_y in loader:
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
            if (epoch + 1) % 10 == 0:
                epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

        return avg_loss


    def predict(self, X: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.eval()
        with torch.no_grad():
            y_hat = self(X_tensor)
        return y_hat.cpu().numpy() # Move back to CPU



class XGBoost(DirectModel):
    def __init__(self, window: int, horizon: int, n_estimators: int, max_depth: int, learning_rate: float, device: str):
        super().__init__(window, horizon)

        self.reg = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            device=device if device == 'cuda' else None
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.reg.fit(X, y)
        return np.mean((self.reg.predict(X) - y) ** 2)

    def predict(self, X: np.ndarray):
        return self.reg.predict(X)



if __name__ == "__main__":
    # Simple test of the TCN model
    window = 24
    horizon = 12
    n_samples = 1000

    # Generate random data
    X = np.random.rand(n_samples, window)
    y = np.random.rand(n_samples, horizon)

    # Initialize and train TCN model
    tcn_model = TCN(
        window=window,
        horizon=horizon,
        n_layers=3,
        inner_layers_dim=[16, 32],
        kernel_size=[3, 3, 3],
        stride=1,
        padding=1,
        activation='relu',
        learning_rate=0.001,
        n_epochs=50,
        device='cpu'
    )

    tcn_model.fit(X, y)
    y_pred = tcn_model.predict(X)

    print("TCN Prediction Shape:", y_pred.shape)

    # Initialize and train XGBoost model
    xgb_model = XGBoost(
        window=window,
        horizon=horizon,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        device='cpu'
    )

    xgb_model.fit(X, y)
    y_pred_xgb = xgb_model.predict(X)

    print("XGBoost Prediction Shape:", y_pred_xgb.shape)

    print("Predictions:", y_pred, y_pred_xgb)