import numpy as np
import torch.nn.functional as F

import torch 

def horizon_weighted_huber(y_hat : np.ndarray | torch.Tensor, y : np.ndarray | torch.Tensor, weights : np.ndarray | torch.Tensor, delta=1.0):
    """

        Compute a horizon-aware Huber loss for direct multi-step (seq2seq) forecasting.

        The loss is computed element-wise using the Huber formulation and then
        weighted along the forecast horizon to allow differential penalization
        of prediction errors at different lead times.

        Params:

        - y_hat : predictions of size horizon
        - y : ground truth of size horizon
        - weights : horizon-wise weighting coefficients of size horizon
        - delta : float, optional. threshold parameter for the Huber loss. 
            For errors with absolute value smaller than `delta`, the loss is quadratic; for larger errors,
            it is linear. Default is 1.0.

        Returns
        
        - The scalar horizon-aware Huber loss, averaged over batch and horizon

    """
    # Ensure tensors
    if not torch.is_tensor(y_hat):
        y_hat = torch.tensor(y_hat, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, dtype=torch.float32)

    weights = weights.to(y_hat.device, y_hat.dtype)

    weights = weights / weights.mean()   

    # elementwise loss: [B, H]
    loss = F.huber_loss(y_hat, y, delta=delta, reduction='none')

    # reshape weights to broadcast: [1, H]
    weights = weights.view(1, -1)

    # apply horizon weights
    loss = loss * weights

    return loss.mean()

def mse(y_hat: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, weights: np.ndarray | torch.Tensor | None = None):
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_hat : array-like, shape (B, H)
        Model predictions.
    y : array-like, shape (B, H)
        Ground-truth values.
    weights : array-like, shape (H,),
        Used only for matching the function signature. Not used in MSE.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss averaged over batch and horizon.
    """

    if not torch.is_tensor(y_hat):
        y_hat = torch.tensor(y_hat, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)

    return F.mse_loss(y_hat, y, reduction="mean")
    
