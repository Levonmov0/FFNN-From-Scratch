import numpy as np

def init_params(layer_sizes):
    """
    Initializes network weights and biases using He initialization.
    
    Args:
        layer_sizes (list[int]): List of layer sizes, including input and output layers.
    
    Returns:
        dict: Dictionary with weight and bias matrices for each layer.
    """
    params = {}
    for i in range(1, len(layer_sizes)):
        input_dim = layer_sizes[i - 1]
        output_dim = layer_sizes[i]
        params["W" + str(i)] = np.random.randn(output_dim, input_dim) * np.sqrt(
            2.0 / input_dim
        )
        params["b" + str(i)] = np.zeros((output_dim, 1))
    return params

def sigmoid(Z):
    """Applies the sigmoid activation function elementwise.
    
    Args:
        Z (ndarray): Pre-activation values.
    
    Returns:
        ndarray: Sigmoid activations.
    """
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    """Computes the derivative of the sigmoid activation.
    
    Args:
        Z (ndarray): Pre-activation values used in the sigmoid.
    
    Returns:
        ndarray: Elementwise derivative of the sigmoid at Z.
    """
    sig = sigmoid(Z)
    return sig * (1 - sig)


def compute_loss(Y_hat, Y):
    """
    Computes the mean squared error loss.
    
    Args:
        Y_hat (ndarray): Predicted outputs of shape (n_outputs, m).
        Y (ndarray): True targets of shape (n_outputs, m).
    
    Returns:
        float: Mean squared error between Y_hat and Y.
    """
    return np.mean((Y_hat - Y) ** 2)


def forward_pass(X, params, layer_sizes):
    """
    Runs the forward propagation through the network.
    
    Args:
        X (ndarray): Input batch of shape (n_features, m).
        params (dict): Network parameters containing weights and biases.
        layer_sizes (list[int]): Architecture of the network.
    
    Returns:
        tuple: (Y_hat, caches) where Y_hat are outputs and caches store intermediates for backprop.
    """
    caches = []
    A0 = X
    for i in range(1, len(layer_sizes)):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        A_prev = A0
        Z = W @ A_prev + b

        if i < len(layer_sizes) - 1:
            A0 = sigmoid(Z)
        else:
            A0 = Z

        caches.append((A_prev, Z, A0))

    Y_hat = A0
    return Y_hat, caches


def backward_pass(Y_hat, Y, caches, params, layer_sizes):
    """
    Runs backpropagation to compute gradients for all parameters.
    
    Args:
        Y_hat (ndarray): Predicted outputs.
        Y (ndarray): True targets.
        caches (list): Cached values from the forward pass.
        params (dict): Current network parameters.
        layer_sizes (list[int]): Network architecture.
    
    Returns:
        dict: Gradients for each parameter keyed by dW/dB names.
    """
    grads = {}
    L = len(layer_sizes) - 1
    m = Y.shape[1]

    dA = Y_hat - Y
    A_prev, Z, _ = caches[-1]
    W = params[f"W{L}"]
    dZ = dA
    grads[f"dW{L}"] = (1 / m) * (dZ @ A_prev.T)
    grads[f"db{L}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    for i in reversed(range(1, L)):
        A_prev, Z, A = caches[i - 1]
        W = params[f"W{i}"]
        dZ = dA_prev * sigmoid_derivative(Z)
        grads[f"dW{i}"] = (1 / m) * (dZ @ A_prev.T)
        grads[f"db{i}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ

    return grads
