import numpy as np

class DirectModel:

    def __init__(self, window : int, horizon : int):
        self.window = window
        self.horizon = horizon

    def fit(self, X : np.ndarray, y : np.ndarray):
        """
            Fit based on the error committed by predicting y_hat using X as input.

            Params : 
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