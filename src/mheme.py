from src.direct_model import *
import numpy as np
from matplotlib import pyplot as plt

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
            The weights are the empirical 

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
    
    def visualize_variances(self, k : int | None = None):
        """
            Method to visualize the variances of each model or the variance of a desired model

            Params : 
            - k : int or None, set to int indicates the index of the desired model to visualize, set to None indicates the whole ensemble as desired model
        """
        if k != None:
            model = self.models[k]
            variances = 1/self.weights[k, :]
            plt.bar([i for i in range (0, self.horizon)], variances)
            plt.title(f"Bar plot for the variances of the errors of the model {model} at each time step")
            plt.xlabel("Time step")
            plt.ylabel("Variance of the errors")
            plt.show()


if __name__ == '__main__':
    pass





