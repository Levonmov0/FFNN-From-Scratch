import numpy as np

def init_adam(layer_sizes):
    """
    Initializes the state dictionary for the Adam optimizer.
    
    Args:
        layer_sizes (list[int]): Network architecture including input and output sizes.
    
    Returns:
        dict: Optimizer state including first and second moment estimates.
    """
    L = len(layer_sizes) - 1
    state = {"t": 0}
    for i in range(1, L + 1):
        state[f"mW{i}"] = 0
        state[f"vW{i}"] = 0
        state[f"mb{i}"] = 0
        state[f"vb{i}"] = 0
    return state


def adam_optimizer(
    params, grads, opt_state, b1=0.9, b2=0.999, learning_rate=1e-3, epsilon=1e-8
):
    """Performs a single Adam optimization step over all parameters.
    
    Args:
        params (dict): Current network parameters.
        grads (dict): Gradients for each parameter.
        opt_state (dict): Adam optimizer state (moments and timestep).
        b1 (float, optional): Exponential decay rate for the first moment. Defaults to 0.9.
        b2 (float, optional): Exponential decay rate for the second moment. Defaults to 0.999.
        learning_rate (float, optional): Step size for parameter updates. Defaults to 1e-3.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
    
    Returns:
        tuple: (params, opt_state) updated parameters and optimizer state.
    """
    
    L = len([k for k in params if k.startswith("W")])  
    opt_state["t"] += 1
    t = opt_state["t"]

    for i in range(1, L + 1):
        # gradients
        gW = grads[f"dW{i}"]
        gb = grads[f"db{i}"]

        # first/second moments
        mW = opt_state[f"mW{i}"] = b1 * opt_state[f"mW{i}"] + (1 - b1) * gW
        vW = opt_state[f"vW{i}"] = b2 * opt_state[f"vW{i}"] + (1 - b2) * (gW * gW)
        mb = opt_state[f"mb{i}"] = b1 * opt_state[f"mb{i}"] + (1 - b1) * gb
        vb = opt_state[f"vb{i}"] = b2 * opt_state[f"vb{i}"] + (1 - b2) * (gb * gb)

        # bias correction
        mW_hat = mW / (1 - b1**t)
        vW_hat = vW / (1 - b2**t)
        mb_hat = mb / (1 - b1**t)
        vb_hat = vb / (1 - b2**t)

        # update params
        params[f"W{i}"] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + epsilon)
        params[f"b{i}"] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

    return params, opt_state
