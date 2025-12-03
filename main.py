from ffnn.data import data_handler
from ffnn.training import train_model
from ffnn.evaluation import evaluate_model
import time
import matplotlib.pyplot as plt

def main():
    # Initializing values
    EPOCHS = 7000
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 32
    DECAY_FACTOR = 0.6
    PATIENCE = 200
    PRINT_EVERY = 150

    start_time = time.time()

    # Load and preprocess data
    X_train, y_train, X_test, y_test, X_val, y_val, y_std = data_handler(
        "data/raw/maintenance.txt"
    )

    layer_sizes = [X_train.shape[0], 25, 25, 2]

    # Train model
    best_params, best_val, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        layer_sizes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        decay_factor=DECAY_FACTOR,
        patience=PATIENCE,
        print_every=PRINT_EVERY,
    )

    # Evaluate model
    _,  _, Y_test_hat = evaluate_model(
        X_test, y_test, best_params, layer_sizes, y_std
    )

    end_time = time.time()
    print(f"Total training and evaluation time: {end_time - start_time:.2f} seconds")

    # Loss Plots
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Learning Rate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history["lr"])
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.show()

    #test plot acctual vs predicted
    plt.figure(figsize=(10, 5))

    # Scatter plot for each output component
    for i in range(Y_test_hat.shape[0]):
        plt.scatter(
            y_test[i],
            Y_test_hat[i],
            alpha=0.6,
            label=f"Output {i}"
        )

        # Diagonal line y=x
        min_val = min(y_test[i].min(), Y_test_hat[i].min())
        max_val = max(y_test[i].max(), Y_test_hat[i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Test Set: Predicted vs Actual")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
