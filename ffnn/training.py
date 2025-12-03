
from ffnn.nn import forward_pass, backward_pass, compute_loss, init_params
from ffnn.optim import init_adam, adam_optimizer
import numpy as np
import copy

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    layer_sizes,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    decay_factor=0.6,
    patience=200,
    print_every=5,
):
    """
    Trains a feedforward neural network using mini-batch gradient descent with Adam optimization.

    Args:
        X_train (ndarray): Training input data of shape (n_features, m_train).
        y_train (ndarray): Training target values of shape (n_outputs, m_train).
        X_val (ndarray): Validation input data.
        y_val (ndarray): Validation target values.
        layer_sizes (list[int]): Network architecture describing units per layer.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Size of each mini-batch. Defaults to 64.
        learning_rate (float, optional): Initial learning rate for Adam. Defaults to 1e-3.
        decay_factor (float, optional): Multiplicative decay applied to learning rate when patience is exceeded. Defaults to 0.6.
        patience (int, optional): Number of epochs without improvement before learning rate decay. Defaults to 200.
        print_every (int, optional): Interval for printing training progress. Defaults to 5.

    Returns:
        tuple: (best_params, best_val_loss, history)
            best_params (dict): Parameter set that achieved the lowest validation loss.
            best_val_loss (float): Best (lowest) recorded validation loss.
            history (dict): Training history containing lists for:
                - "train_loss": Mean training loss per epoch.
                - "val_loss": Validation loss per epoch.
                - "lr": Learning rate per epoch.
    """

    params = init_params(layer_sizes)
    opt_state = init_adam(layer_sizes)

    best_params = copy.deepcopy(params)
    best_val_loss = float("inf")

    patience_counter = 0

    # Initialize history dictionary
    history = {"train_loss": [], "val_loss": [], "lr": []}

    def iterate_minibatches(X, Y, batch_size=64, shuffle=True):
        """
        Generates mini-batches from the input and target datasets.

        Args:
            X (ndarray): Input data of shape (n_features, m_samples).
            Y (ndarray): Target data of shape (n_outputs, m_samples).
            batch_size (int, optional): Number of samples per mini-batch. Defaults to 64.
            shuffle (bool, optional): Whether to shuffle the sample indices before batching. Defaults to True.

        Yields:
            tuple: (X_batch, Y_batch)
                X_batch (ndarray): Mini-batch input slice of shape (n_features, batch_size).
                Y_batch (ndarray): Mini-batch target slice of shape (n_outputs, batch_size).
        """

        m = X.shape[1]
        idx = np.arange(m)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, m, batch_size):
            b = idx[start : start + batch_size]
            yield X[:, b], Y[:, b]

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for Xb, Yb in iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            Y_hat, caches = forward_pass(Xb, params, layer_sizes)
            loss = compute_loss(Y_hat, Yb)
            grads = backward_pass(Y_hat, Yb, caches, params, layer_sizes)
            params, opt_state = adam_optimizer(params, grads, opt_state, learning_rate=learning_rate)
            epoch_losses.append(loss)

        train_loss_mean = float(np.mean(epoch_losses))

        # validation
        Y_val_hat, _ = forward_pass(X_val, params, layer_sizes)
        val_loss = compute_loss(Y_val_hat, y_val)

        # track improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = copy.deepcopy(params)
            patience_counter = 0
        else:
            patience_counter += 1

        # LR decay
        if patience_counter >= patience:
            learning_rate *= decay_factor
            patience_counter = 0
            print(f"Learning rate decayed to {learning_rate}")

        #Store history
        history["train_loss"].append(train_loss_mean)
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(learning_rate))

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss_mean:.6f} | val_loss={val_loss:.6f} "
                f"| best_val={best_val_loss:.6f} | lr={learning_rate:.6f}"
            )

    # NEW: return history always
    return best_params, best_val_loss, history
