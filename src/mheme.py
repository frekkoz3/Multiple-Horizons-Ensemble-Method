from src.direct_models import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import pickle
import os


def autoregressive_prediction(model : DirectModel, total_horizon : int, X : np.ndarray):
    """
        Perform block autoregressive prediction using the full model horizon at each step.

        Params : 
        - model : DirectModel instance
        - total_horizon : desired total prediction horizon
        - X : input of shape set_size x window_size

        Return :
        - y_hat : output of shape set_size x total_horizon
    """
    window_size = X.shape[1]

    X_curr = X.copy()
    y_hat = []

    while sum(y.shape[1] for y in y_hat) < total_horizon:
        y_block = model.predict(X_curr) # set_size x model_horizon
        # y_block = np.atleast_2d(y_block) # ensure it is 2-dimensional
        y_block = y_block.reshape(X_curr.shape[0], -1) # set_size x model_horizon

        y_hat.append(y_block) # append the predicted block to the final prediction

        # Update X with the newest predictions and remove what it is not necessary anymore
        X_curr = np.concatenate(
            [X_curr, y_block],
            axis=1
        )[:, -window_size:]

    # Concatenate all predicted blocks
    y_hat = np.concatenate(y_hat, axis=1)

    return y_hat[:, :total_horizon]


class UMHEMe:

    def __init__(self, horizon, window, model_class : DirectModel, config_path : str):
        self.horizon = horizon
        self.window = window
        self.models = [model_class(h, config_path) for h in range(1, horizon + 1)]
        self.weights = np.ones(shape = (self.horizon, self.horizon)) # n_models x horizon

    def fit(self, X, y):
        """
            Fit the whole ensemble method. Passes a set of X and corresponding ground truth. 

            Params : 
            - X : input of shape set_size x self.window 
            - y : ground truth of shape set_size x self.horizon

            Note : 
            - Before fitting the single models it must be ensured that y will be shrink down to the right shape, since each model has a different horizon and each model demands a ground truth of the correct shape
        """
        for i, m in enumerate(self.models):
            h = i+1
            y_h = y[:, :h] 
            m.fit(X, y_h)
        
    def compute_weights(self, X, y):
        """
            Compute the weights for each model prediction's as described in the "Univariate MHEMe.md" file.
            The weights are the empirical variances of the errors committed by each model during prediction.

            Params:
            - X : input of shape set_size x self.window
            - y : ground truth of shape set_size x self.horizon
        """
        y_hat = np.array([autoregressive_prediction(m, self.horizon, X) for m in self.models]) # n_models x set_size x total_horizon
        errors = y_hat - y[None, : , : ] # n_models x set_size x total_horizon
        error_var = errors.var(axis=1)        # n_models x total_horizon
        eps = 1e-8 # numerical stability 
        self.weights = 1.0 / (error_var + eps) # n_models x total_horizon 

    def predict(self, X):
        """
            Predict y_hat using X as input. 

            Params : 
            - X : input of shape set_size x .window 

            Return:
            - y_hat : output of shape set_size x self.horizon
        """
        y_hat = np.array([autoregressive_prediction(m, self.horizon, X) for m in self.models]) # n_models x set_size x total_horizon
        # expand weights to broadcast over set_size
        w = self.weights[:, None, :]          # n_models x 1 x total_horizon

        # weighted sum over models
        num = (y_hat * w).sum(axis=0)    # set_size x total_horizon
        den = w.sum(axis=0)              # 1 x total_horizon

        y_combined = num / den # set_size x total_horizon

        return y_combined
    
    def whole_predict(self, X):
        """
            Predict y_hat using X as input. Returns all the predictions from all the models in the ensemble. 

            Params : 
            - X : input of shape set_size x .window 

            Return:
            - dictionary where the key are the models' names and the values are the models' predictions, shape n_models x total_horizon
        """
        y_hat = np.array([autoregressive_prediction(m, self.horizon, X) for m in self.models]) # n_models x set_size x total_horizon
        return {str(m) : y_hat for m in self.models}
    
    def visualize_variances(self, k: int | None = None):
        """
        Visualize the variances of each model or a desired model using Plotly.

        Params:
        - k : int or None
            If int, visualize only the model at index k.
            If None, visualize all models in the ensemble.
        """
        time_steps = np.arange(self.horizon)

        if k is not None:
            model = self.models[k]
            variances = 1 / self.weights[k, :]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=variances,
                mode='lines+markers',
                line=dict(color='red', width=2),
                name=f"Model {model.horizon}"
            ))
            fig.update_layout(
                title=f"Variance of prediction errors for model {model.horizon} at each time step",
                xaxis_title="Time step",
                yaxis_title="Variance of errors"
            )
            fig.show()

        else:
            fig = go.Figure()
            n_models = len(self.models)
            # Create a color gradient from red to blue
            colors = px.colors.sample_colorscale("RdBu", [i/(n_models-1) for i in range(n_models)])

            for idx, model in enumerate(self.models):
                variances = 1 / self.weights[idx, :]
                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=variances,
                    mode='lines+markers',
                    line=dict(color=colors[idx], width=2),
                    name=f"Model horizon {model.horizon}"
                ))

            fig.update_layout(
                title="Variance of prediction errors for each model at each time step",
                xaxis_title="Time step",
                yaxis_title="Variance of errors",
                legend_title="Models"
            )
            fig.show()

    def visualize_weights(self, k : int | None = None):
        """
        Visualize the weights of each model or a desired model using Plotly.

        Params:
        - k : int or None
            If int, visualize only the model at index k.
            If None, visualize all models in the ensemble.
        """
        time_steps = np.arange(self.horizon)

        if k is not None:
            model = self.models[k]
            w = self.weights/ self.weights.sum(axis = 0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=w[k, :],
                mode='lines+markers',
                line=dict(color='red', width=2),
                name=f"Model {model.horizon}"
            ))
            fig.update_layout(
                title=f"Weights of prediction errors for model {model.horizon} at each time step",
                xaxis_title="Time step",
                yaxis_title="Weights"
            )
            fig.show()

        else:
            fig = go.Figure()
            n_models = len(self.models)
            # Create a color gradient from red to blue
            colors = px.colors.sample_colorscale("RdBu", [i/(n_models-1) for i in range(n_models)])
            w = self.weights/ self.weights.sum(axis = 0)
            for idx, model in enumerate(self.models):
                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=w[idx, :],
                    mode='lines+markers',
                    line=dict(color=colors[idx], width=2),
                    name=f"Model horizon {model.horizon}"
                ))

            fig.update_layout(
                title="Weights of prediction errors for each model at each time step",
                xaxis_title="Time step",
                yaxis_title="Weights",
                legend_title="Models"
            )
            fig.show()

    def save_model(self, model_path : str):
        """
        Save the whole ensemble model to disk.

        Params:
        - model_path : pickle path (path_to_model.pkl) where to save the model
        """
        assert model_path.endswith('.pkl'), "Model path must end with .pkl"

        directory = os.path.dirname(model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        return


    @staticmethod
    def load_model(model_path : str):
        """
        Load the whole ensemble model from disk.

        Params:
        - model_path : pickle path (path_to_model.pkl) from where to load the model

        Returns:
        - UMHEMe instance
        """
        assert model_path.endswith('.pkl'), "Model must be a valid pickle file"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model


    @staticmethod
    def model_loss(y_true : np.ndarray, y_pred : np.ndarray) -> float:
        """
        Simply computing RMSE given a ground truth time series and prediction time series

        Params:
        - y_true : array of shape (horizon, ) containing ground truth values
        - y_pred : array of shape (horizon, ) containing predicted values

        Returns:
        - rmse : float value representing the RMSE between y_true and y_pred
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

if __name__ == '__main__':
    pass





