from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from nn_l2_full_updated import predict, initialize_parameters, compute_cost, update_parameters, \
    l_model_forward, l_model_backward

def l_layer_model(X, Y, X_val, Y_val, layers_dims, learning_rate,
                  max_training_steps, batch_size, max_no_improve_steps=100,
                  use_batchnorm=False, lambd=0.0):
    parameters = initialize_parameters(layers_dims)
    val_costs = []
    train_costs = []

    m = X.shape[1]
    step = 0
    val_check_count = 0
    best_val_cost = float('inf')
    no_improve_counts = 0

    print("Starting training...")

    while step < max_training_steps and no_improve_counts < max_no_improve_steps:
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for j in range(0, m, batch_size):
            if step >= max_training_steps or no_improve_counts >= max_no_improve_steps:
                break

            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[:, j:j + batch_size]

            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm)
            grads = l_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate, lambd)

            if step % 100 == 0:
                train_cost = compute_cost(AL, Y_batch, parameters, lambd)
                train_costs.append((step, train_cost))

                AL_val, _ = l_model_forward(X_val, parameters, use_batchnorm)
                val_cost = compute_cost(AL_val, Y_val, parameters, lambd)
                val_costs.append((step, val_cost))

                print(f"Step {step}: Train cost = {train_cost:.4f} | Validation cost = {val_cost:.4f}")

                if val_cost + 1e-5 < best_val_cost:
                    best_val_cost = val_cost
                    no_improve_counts = 0
                else:
                    no_improve_counts += 1

                val_check_count += 1

            step += 1

    print("Training complete!")
    print(f"Total training steps: {step}")
    print(f"Total validation evaluations: {val_check_count}")
    print(f"Best validation cost: {best_val_cost:.4f}")

    return parameters, val_costs, train_costs

if __name__ == '__main__':
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009
    batch_size = 128
    max_training_steps = 100000
    max_no_improve_steps = 100
    use_batchnorm = False
    lambd = 0.0001  # L2 regularization strength

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1).T / 255.0
    test_X = test_X.reshape(test_X.shape[0], -1).T / 255.0
    train_y = to_categorical(train_y).T
    test_y = to_categorical(test_y).T

    m_train = train_X.shape[1]
    perm = np.random.permutation(m_train)
    val_size = int(0.2 * m_train)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    X_train = train_X[:, train_idx]
    Y_train = train_y[:, train_idx]
    X_val = train_X[:, val_idx]
    Y_val = train_y[:, val_idx]

    parameters, val_costs, train_costs = l_layer_model(
        X_train, Y_train, X_val, Y_val, layers_dims,
        learning_rate, max_training_steps, batch_size,
        max_no_improve_steps=max_no_improve_steps,
        use_batchnorm=use_batchnorm,
        lambd=lambd
    )

    train_acc = predict(X_train, Y_train, parameters, use_batchnorm)
    val_acc = predict(X_val, Y_val, parameters, use_batchnorm)
    test_acc = predict(test_X, test_y, parameters, use_batchnorm)

    print(f"\nFinal Train Accuracy: {train_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    steps_val, val_costs_only = zip(*val_costs)
    steps_train, train_costs_only = zip(*train_costs)

    plt.plot(steps_train, train_costs_only, label='Training Cost', linestyle='--')
    plt.plot(steps_val, val_costs_only, label='Validation Cost', marker='o')
    plt.xlabel('Training Step')
    plt.ylabel('Cost')
    plt.title('Training & Validation Cost with L2 Regularization')
    plt.grid(True)
    plt.legend()
    plt.savefig("training_validation_costs_l2_no_batch_norm.png")
    plt.show()

    print("\nTraining and Validation Cost every 100 steps:")
    for (s_t, c_t), (s_v, c_v) in zip(train_costs, val_costs):
        print(f"Step {s_t:5d} - Train Cost: {c_t:.4f} | Val Cost: {c_v:.4f}")

    # Print L2 norm of weights for analysis
    print("\nL2 Norm of Weights by Layer:")
    for l in range(1, len(layers_dims)):
        W = parameters[f"W{l}"]
        norm = np.linalg.norm(W)
        print(f"W{l} norm: {norm:.4f}")
