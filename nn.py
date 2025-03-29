from typing import List, Tuple, Dict

import numpy as np


def initialize_parameters(layer_dims: List[int]) -> Dict:
    """

    Args:
        layer_dims: an array of the dimensions of each layer in the network
         (layer 0 is the size of the flattened input, layer L is the output softmax)

    Returns:
        A dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters[f"W{i}"] = np.random.rand(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))

    return parameters


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Implement the linear part of a layer's forward propagation.
    Args:
        A: The activations of the previous layer
        W: The weight matrix of the current layer (of shape [size of current layer, size of previous layer])
        b: The bias vector of the current layer (of shape [size of current layer, 1])

    Returns:
        Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
        linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)

    """
    return np.dot(W, A) + b, {'A': A, 'W': W, 'b': b}


def softmax(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """

    Args:
        Z: The linear component of the activation function

    Returns:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation

    """
    shift_z = np.max(Z, axis=0, keepdims=True)
    exp_z = np.exp(Z - shift_z)
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return A, {'Z': Z}


def relu(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """

    Args:
        Z: The linear component of the activation function

    Returns:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation

     """
    A = np.maximum(Z, 0)
    return A, {'Z': Z}


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, B: np.ndarray, activation: str) -> Tuple[
    np.ndarray, Dict]:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Args:
        A_prev: activations of the previous layer
        W: the weights matrix of the current layer
        B: the bias vector of the current layer
        activation: the activation function to be used (a string, either “softmax” or “relu”)

    Returns:
        A – the activations of the current layer
        cache – a joint dictionary containing both linear_cache and activation_cache

    """
    activations = {
        'relu': relu,
        'softmax': softmax
    }
    activation = activations.get(activation)
    if activation is None:
        raise ValueError(f"Only the following functions are available: {activations.keys()}")

    Z, linear_cache = linear_forward(A_prev, W, B)
    A, activation_cache = activation(Z)

    return A, {**linear_cache, **activation_cache}


def l_model_forward(X: np.ndarray, parameters: Dict, use_batchnorm: bool) -> Tuple[np.ndarray, List[Dict]]:
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    Args:
        X: the data, numpy array of shape (input size, number of examples)

        parameters: the initialized W and b parameters of each layer
        use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation

    Returns:
        AL – the last post-activation value
        caches – a list of all the cache objects generated by the linear_forward function

    """
    A = X
    caches = []
    L = len(parameters) // 2  # number of layers

    # Loop through L-1 hidden layers with ReLU
    for i in range(1, L):
        W = parameters[f'W{i}']
        b = parameters[f'b{i}']
        A, cache = linear_activation_forward(A, W, b, activation='relu')
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    # Final layer with softmax
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    AL, cache = linear_activation_forward(A, WL, bL, activation='softmax')
    caches.append(cache)

    return AL, caches


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    """
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss
    Args:
        AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
        Y: the labels vector (i.e. the ground truth)

    Returns:
        cost – the cross-entropy cost
    """
    m = Y.shape[1]
    AL = np.clip(AL, 1e-12, 1. - 1e-12)

    cost = -np.sum(Y * np.log(AL)) / m

    return cost


def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    """
    performs batchnorm on the received activation values of a given layer.
    Args:
        A:the activation values of a given layer

    Returns:
        NA - the normalized activation values, based on the formula learned in class
    """
    epsilon = 1e-8
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    return (A - mean) / np.sqrt(var + epsilon)


def linear_backward(dZ: np.ndarray, cache: Dict) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements the linear part of the backward propagation process for a single layer
    Args:
        dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
        cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache["A"], cache["W"], cache["b"]
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(b, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA: np.ndarray, cache: Dict, activation: str):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
    The function first computes dZ and then applies the linear_backward function.
    Args:
        dA:post activation gradient of the current layer
        cache: contains both the linear cache and the activations cache
        activation: str func

    Returns:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b

    """
    activations = {
        'relu': relu_backward,
        'softmax': softmax_backward
    }
    activation = activations.get(activation)
    if activation is None:
        raise ValueError(f"Only the following functions are available: {activations.keys()}")

    dz = activation(dA, cache)
    return linear_backward(dz, cache)


def relu_backward(dA, activation_cache) -> np.ndarray:
    """
    Implements backward propagation for a ReLU unit
    Args:
        dA: the post-activation gradient
        activation_cache: contains Z (stored during the forward propagation)

    Returns:
        dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache["Z"]
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dA


def softmax_backward(dA, activation_cache):
    """
    Implements backward propagation for a softmax unit
    Args:
        dA: the post-activation gradient
        activation_cache: contains Z (stored during the forward propagation)

    Returns:
        dZ – gradient of the cost with respect to Z

    """
    return dA


def l_model_backward(AL: np.ndarray, Y: np.ndarray, caches: List):
    """
    Implement the backward propagation process for the entire network.
    Args:
        AL: the probabilities vector, the output of the forward propagation (L_model_forward)
        Y: the true labels vector (the "ground truth" - true classifications)
        caches: list of caches containing for each layer: a) the linear cache; b) the activation cache

    Returns:
        Grads - a dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...

    Notes:
        the backpropagation for the softmax function should be done only once as only the
         output layers uses it and the RELU should be done iteratively over all the remaining layers of the network.
    """
    grads = {}
    L = len(caches)
    dz = AL - Y
    cache = caches[-1]
    dA_prev, dW, db = linear_backward(dz, cache)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(L - 1)):
        current_cache = caches[l]

        dA = grads["dA" + str(l + 1)]
        dZ = relu_backward(dA, current_cache)
        dA_prev, dW, db = linear_backward(dZ, current_cache)

        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    Args:
        parameters: a python dictionary containing the DNN architecture’s parameters
        grads: a python dictionary containing the gradients (generated by L_model_backward)
        learning_rate: the learning rate used to update the parameters (the “alpha”)

    Returns:
        parameters – the updated values of the parameters object provided as input
    """
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters
