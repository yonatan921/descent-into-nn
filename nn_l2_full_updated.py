
from typing import List, Tuple, Dict
import numpy as np

def initialize_parameters(layer_dims: List[int]) -> Dict:
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters[f"W{i}"] = np.random.rand(layer_dims[i], layer_dims[i - 1]) * 0.01
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
    activations = {'relu': relu, 'softmax': softmax}
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
    return AL, caches

def compute_cost(AL: np.ndarray, Y: np.ndarray, parameters: Dict = None, lambd: float = 0.0) -> float:
    m = Y.shape[1]
    AL = np.clip(AL, 1e-12, 1. - 1e-12)
    cross_entropy_cost = -np.sum(Y * np.log(AL)) / m
    l2_cost = 0
    if parameters and lambd > 0:
        for key in parameters:
            if key.startswith("W"):
                l2_cost += np.sum(np.square(parameters[key]))
        l2_cost = (lambd / (2 * m)) * l2_cost
    return cross_entropy_cost + l2_cost

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

def relu_backward(dA, activation_cache) -> np.ndarray:
    Z = activation_cache["Z"]
    return dA * (Z > 0)

def softmax_backward(dA, activation_cache):
    Y = activation_cache['Y']
    return dA - Y

def linear_activation_backward(dA: np.ndarray, cache: Dict, activation: str):
    activations = {'relu': relu_backward, 'softmax': softmax_backward}
    activation = activations.get(activation)
    if activation is None:
        raise ValueError(f"Only the following functions are available: {activations.keys()}")
    dz = activation(dA, cache)
    return linear_backward(dz, cache)

def l_model_backward(AL: np.ndarray, Y: np.ndarray, caches: List):
    grads = {}
    L = len(caches)
    cache = caches[-1]
    cache['Y'] = Y
    dA_prev, dW, db = linear_activation_backward(AL, cache, activation='softmax')
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    for l in reversed(range(L - 1)):
        cache = caches[l]
        dA = grads["dA" + str(l + 1)]
        dA_prev, dW, db = linear_activation_backward(dA, cache, activation='relu')
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    return grads

def update_parameters(parameters, grads, learning_rate, lambd=0.0):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        W_key = f"W{l}"
        b_key = f"b{l}"
        if lambd > 0:
            grads[f"dW{l}"] += (lambd / grads[f"dW{l}"].shape[1]) * parameters[W_key]
        parameters[W_key] -= learning_rate * grads[f"dW{l}"]
        parameters[b_key] -= learning_rate * grads[f"db{l}"]
    return parameters

def predict(X: np.ndarray, Y: np.ndarray, parameters: Dict, use_batchnorm: bool = False) -> float:
    AL, _ = l_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)
    return np.mean(predictions == true_labels) * 100
