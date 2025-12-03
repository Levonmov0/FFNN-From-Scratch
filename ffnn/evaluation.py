
from ffnn.nn import forward_pass, compute_loss
import numpy as np

def evaluate_model(X_test, y_test, params, layer_sizes, y_std):
    """
    Evaluates the trained model on a test set and reports MSE metrics.
    
    Args:
        X_test (ndarray): Test inputs.
        y_test (ndarray): Test targets (normalized).
        params (dict): Trained network parameters.
        layer_sizes (list[int]): Network architecture.
        y_std (ndarray): Standard deviation used to denormalize outputs.
    
    Returns:
        float: Mean squared error on the test set in original scale.
    """
    
    Y_hat, _ = forward_pass(X_test, params, layer_sizes)

    # overall MSE (both outputs together)
    test_loss = compute_loss(Y_hat, y_test)

    # per-output MSE (comp = first output, turb = second output)
    se = (Y_hat - y_test) ** 2
    mse_per_output = np.mean(se, axis=1)
    mse_per_output_original = mse_per_output * (y_std.flatten() ** 2)

    print(f"Test MSE (overall): {test_loss:.6f}")
    print(f"Compressor MSE: {mse_per_output_original[0]}")
    print(f"Turbine MSE:    {mse_per_output_original[1]}")

    return test_loss, mse_per_output_original, Y_hat