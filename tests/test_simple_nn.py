import numpy as np

from advanced_precision.simple_nn import forward


def test_forward_tiny_network():
    weights = [
        np.array([[0.2, -0.4], [0.7, 0.1]], dtype=float),
        np.array([[0.5], [-0.3]], dtype=float),
    ]
    biases = [
        np.array([0.1, -0.2], dtype=float),
        np.array([0.05], dtype=float),
    ]
    inputs = np.array([[1.0, 0.5]], dtype=float)

    hidden = np.tanh(inputs @ weights[0] + biases[0])
    expected = hidden @ weights[1] + biases[1]

    output = forward(weights, biases, inputs)

    assert output.shape == expected.shape
    assert abs(output - expected).max() < 1e-12
