from typing import List, Tuple, Dict
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from nn import l_model_forward, relu_backward, l_layer_model, predict, compute_cost

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth if needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is being used.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

def analyze_output(AL: np.ndarray):
    print("\nSoftmax Output Analysis")
    print("Shape:", AL.shape)
    print("Mean:", np.mean(AL))
    print("Std:", np.std(AL))
    print("Sum per sample (first 10):", np.sum(AL, axis=0)[:10])
    if np.any(np.isnan(AL)):
        print("\u26a0\ufe0f Warning: Found NaNs in AL")
    if np.any(np.sum(AL, axis=0) > 1.01) or np.any(np.sum(AL, axis=0) < 0.99):
        print("\u26a0\ufe0f Warning: Some softmax outputs do not sum to ~1")


def analyze_input(X: np.ndarray, Y: np.ndarray):
    print("\nInput Analysis")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Y[:, 0]:", Y[:, 0])
    print("Label (argmax of Y[:,0]):", np.argmax(Y[:, 0]))


def plot_costs(costs: List[float]):
    plt.plot(np.arange(100, 100 * len(costs) + 1, 100), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost over Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_relu_backward():
    Z = np.array([[-1.0, 0.0, 2.0]])
    dA = np.array([[1.0, 2.0, 3.0]])
    cache = {"Z": Z}
    dZ = relu_backward(dA, cache)
    print("\nReLU Backward Test:")
    print("Z:", Z)
    print("dA:", dA)
    print("dZ:", dZ)


def sanity_check_parameters(parameters: Dict):
    print("\nSanity Check on Parameters")
    for k, v in parameters.items():
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            print(f"\u26a0\ufe0f Warning: {k} contains NaNs or Infs")
        else:
            print(f"{k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}, shape={v.shape}")


if __name__ == '__main__':

    layers_dims = [784, 128, 64, 32, 10]
    learning_rate = 0.009
    batch_size = 128
    max_no_improve_steps = 100

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1).T / 255.0
    test_X = test_X.reshape(test_X.shape[0], -1).T / 255.0
    train_y = to_categorical(train_y).T
    test_y = to_categorical(test_y).T

    parameters, cost = l_layer_model(train_X, train_y, layers_dims, learning_rate, 100, batch_size)
    test_acc = predict(test_X, test_y, parameters)
    print(f"{test_acc=}")
    exit(0)


    m_train = train_X.shape[1]
    perm = np.random.permutation(m_train)
    val_size = int(0.2 * m_train)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    X_train = train_X[:, train_idx]
    Y_train = train_y[:, train_idx]
    X_val = train_X[:, val_idx]
    Y_val = train_y[:, val_idx]

    parameters = None
    best_val_acc = 0
    no_improve_count = 0
    iterations = 0
    epoch = 0
    cost_log = []
    m = X_train.shape[1]
    num_batches = m // batch_size + int(m % batch_size != 0)

    print("Starting training...")

    while no_improve_count < max_no_improve_steps:
        epoch += 1
        permutation = np.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[:, permutation]
        epoch_cost = 0

        for j in range(0, m, batch_size):
            iterations += 1
            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[:, j:j + batch_size]

            parameters, _ = l_layer_model(X_batch, Y_batch, layers_dims, learning_rate, 1, batch_size)

            AL, _ = l_model_forward(X_batch, parameters, use_batchnorm=True)
            batch_cost = compute_cost(AL, Y_batch)
            epoch_cost += batch_cost

            if iterations % 100 == 0:
                avg_cost = epoch_cost / num_batches
                cost_log.append(avg_cost)
                val_acc = predict(X_val, Y_val, parameters)
                print(f"Iteration {iterations}: Val Acc = {val_acc:.2f}%, Cost = {avg_cost:.4f}")

                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    no_improve_count = 0
                else:
                    no_improve_count += 1

    train_acc = predict(X_train, Y_train, parameters)
    val_acc = predict(X_val, Y_val, parameters)
    test_acc = predict(test_X, test_y, parameters)

    print("\nFinal Report:")
    print(f"Epochs: {epoch}")
    print(f"Iterations: {iterations}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\nCost values (every 100 steps):")
    for i, c in enumerate(cost_log):
        print(f"Step {(i + 1) * 100}: Cost = {c:.4f}")

    plot_costs(cost_log)
    test_relu_backward()
