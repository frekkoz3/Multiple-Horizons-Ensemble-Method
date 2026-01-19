from src.direct_model import *

class UMHEMe:

    def __init__(self, window : int, horizon : int, model_class : DirectModel):
        self.horizon = horizon
        self.window = window
        self.models = [model_class(window, h) for h in range(1, horizon + 1)]

    def fit(self, X, y):
        """
            Fit based on the error committed by predicting y_hat using X as input.

            Params : 
            - X : input of size self.window 
            - y : ground truth of size self.horizon

            Returns : 
            - eps : errors committed during the prediction
        """

