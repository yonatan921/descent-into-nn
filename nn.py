from typing import List, Tuple, Dict

import numpy as np

def initialize_parameters(layer_dims: List[int]) -> Dict:
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))
    return parameters

def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
    return np.dot(W, A) + b, {'A': A, 'W': W, 'b': b}

def softmax(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    shift_z = np.max(Z, axis=0, keepdims=True)
    exp_z = np.exp(Z - shift_z)
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return A, {'Z': Z}

def relu(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    A = np.maximum(Z, 0)
    return A, {'Z': Z}

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, B: np.ndarray, activation: str) -> Tuple[np.ndarray, Dict]:
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
    A = X
    caches = []
    L = len(parameters) // 2
    for i in range(1, L):
        W = parameters[f'W{i}']
        b = parameters[f'b{i}']
        A, cache = linear_activation_forward(A, W, b, activation='relu')
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    AL, cache = linear_activation_forward(A, WL, bL, activation='softmax')
    caches.append(cache)

    # DEBUGGING BLOCK
    # print("\nDebug Forward Propagation:")
    # print("AL shape:", AL.shape)
    # print("AL mean:", np.mean(AL))
    # print("AL sum per sample (first 5):", np.sum(AL, axis=0)[:5])

    return AL, caches

def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    m = Y.shape[1]
    AL = np.clip(AL, 1e-12, 1. - 1e-12)
    cost = -np.sum(Y * np.log(AL)) / m
    return cost

def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    return (A - mean) / np.sqrt(var + epsilon)

def linear_backward(dZ: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_prev, W, b = cache["A"], cache["W"], cache["b"]
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA: np.ndarray, cache: Dict, activation: str):
    activations = {
        'relu': relu_backward
    }
    if activation == 'softmax':
        raise ValueError("Use AL - Y directly for softmax gradient")
    activation = activations.get(activation)
    if activation is None:
        raise ValueError(f"Only the following functions are available: {activations.keys()}")
    dz = activation(dA, cache)
    return linear_backward(dz, cache)

def relu_backward(dA, activation_cache) -> np.ndarray:
    Z = activation_cache["Z"]
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def l_model_backward(AL: np.ndarray, Y: np.ndarray, caches: List):
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
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    parameters = initialize_parameters(layers_dims)
    costs = []
    m = X.shape[1]
    num_batches = m // batch_size + int(m % batch_size != 0)

    # print("\nDebug Input: X shape:", X.shape, "Y shape:", Y.shape)
    # print("Example label vector (Y[:,0]):", Y[:, 0])

    for i in range(1, num_iterations + 1):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        epoch_cost = 0
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[:, j:j + batch_size]
            AL, caches = l_model_forward(X_batch, parameters, False)
            epoch_cost += compute_cost(AL, Y_batch)
            grads = l_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            costs.append(epoch_cost / num_batches)
    return parameters, costs

def predict(X: np.ndarray, Y: np.ndarray, parameters: Dict):
    AL, _ = l_model_forward(X, parameters, use_batchnorm=False)
    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy
