from nn import *
from pprint import pprint


def main():

    # Dummy input and one-hot labels (2 examples)
    np.random.seed(95)
    X = np.random.randn(4, 2)
    Y = np.array([[1, 0], [0, 1]])

    # Model configuration
    layer_dims = [4, 3, 2]
    learning_rate = 0.00001
    num_iterations = 100

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        # Forward
        AL, caches = l_model_forward(X, parameters, use_batchnorm=False)
        print(f'{AL=}')
        # Cost
        cost = compute_cost(AL, Y)

        # Backward
        grads = l_model_backward(AL, Y, caches)

        # Update
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print cost every 10 iterations
        if i % 10 == 0 or i == num_iterations - 1:
            print(f"Iteration {i} - Cost: {cost:.6f}")

    # Show improvement in W1
    print("\nFinal W1:\n", parameters["W1"])


if __name__ == '__main__':
    main()
