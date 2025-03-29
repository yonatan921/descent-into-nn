from nn import *
from pprint import pprint


def main():
    # Define the architecture: 4 input features, two hidden layers with 3 and 2 units, and 4 output classes
    layer_dims = [4, 3, 2, 4]

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Create sample input data: 4 features (rows) and 5 examples (columns)
    np.random.seed(0)  # For reproducibility
    X_sample = np.random.randn(4, 5)

    # Perform forward propagation
    AL, caches = l_model_forward(X_sample, parameters, use_batchnorm=True)

    # Display the output
    print("Output of the final layer (AL):")
    pprint(AL)

    print("\nNumber of caches stored:")
    print(len(caches))
    for ca in caches:
        print(ca)


if __name__ == '__main__':
    main()
