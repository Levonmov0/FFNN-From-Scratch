import numpy as np
import pandas as pd

def data_handler(pdf_path: str):
    """
    Loads and normalizes the dataset, and splits it into training, validation and test sets.
    
    Args:
        pdf_path (str): Path to the whitespace-separated data file.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, X_validation, y_validation, y_std).
    """
    data = pd.read_csv(pdf_path, sep=r"\s+", header=None)

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    test_size = int(0.25 * m)
    validation_size = int(0.25 * m)
    train_size = m - test_size - validation_size

    data_train = data[0:train_size]
    data_test = data[train_size : train_size + test_size]
    data_validation = data[train_size + test_size : m]

    # Split features and targets
    X_train, y_train = data_train[:, :16], data_train[:, 16:]
    X_test, y_test = data_test[:, :16], data_test[:, 16:]
    X_validation, y_validation = data_validation[:, :16], data_validation[:, 16:]

    # Transpose to shape (features, samples)
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    X_validation = X_validation.T
    y_validation = y_validation.T

    # Normalize using training set statistics
    X_mean = X_train.mean(axis=1, keepdims=True)
    X_std = X_train.std(axis=1, keepdims=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    X_validation = (X_validation - X_mean) / X_std

    y_mean = y_train.mean(axis=1, keepdims=True)
    y_std = y_train.std(axis=1, keepdims=True) + 1e-8  # add small number to avoid div/0
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    y_validation = (y_validation - y_mean) / y_std

    return X_train, y_train, X_test, y_test, X_validation, y_validation, y_std