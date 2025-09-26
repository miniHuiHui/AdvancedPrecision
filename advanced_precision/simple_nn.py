"""A tiny NumPy-based feed-forward neural network helper."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


def _validate_layers(weights: Sequence[Array], biases: Sequence[Array]) -> None:
    """Validate that the provided layer parameters are compatible."""
    if len(weights) != len(biases):
        raise ValueError("weights and biases must contain the same number of layers")

    for idx, (w, b) in enumerate(zip(weights, biases)):
        if w.ndim != 2:
            raise ValueError(f"weight matrix at index {idx} must be 2D, got {w.ndim}D")
        if b.ndim != 1:
            raise ValueError(f"bias vector at index {idx} must be 1D, got {b.ndim}D")
        if w.shape[1] != b.shape[0]:
            raise ValueError(
                "bias vector length must match the output dimension of the weight matrix"
            )

    for prev, curr in zip(weights, weights[1:]):
        if prev.shape[1] != curr.shape[0]:
            raise ValueError("adjacent weight matrices have incompatible shapes")


def forward(
    weights: Sequence[Array],
    biases: Sequence[Array],
    inputs: Array,
    *,
    hidden_activation: Iterable[str] | None = None,
) -> Array:
    """Compute the forward pass of a simple feed-forward neural network.

    Parameters
    ----------
    weights:
        Sequence of weight matrices ``(in_features, out_features)`` for each layer.
    biases:
        Sequence of bias vectors ``(out_features,)`` corresponding to ``weights``.
    inputs:
        Two-dimensional array of shape ``(batch, in_features)`` containing the
        input activations.
    hidden_activation:
        Optional iterable describing the activation used for each hidden layer.
        The supported values are ``"tanh"``, ``"relu"``, and ``"identity"``.
        When omitted, ``tanh`` is used for all hidden layers and the final
        layer remains linear.

    Returns
    -------
    numpy.ndarray
        The network output after applying the linear layers and activations.
    """

    _validate_layers(weights, biases)

    if inputs.ndim != 2:
        raise ValueError("inputs must be a 2D array of shape (batch, features)")
    if inputs.shape[1] != weights[0].shape[0]:
        raise ValueError("input features do not match the first layer dimensions")

    if hidden_activation is None:
        activations = ["tanh"] * (len(weights) - 1)
    else:
        activations = list(hidden_activation)
        if len(activations) not in {len(weights) - 1, len(weights)}:
            raise ValueError(
                "hidden_activation must specify per-layer activations (excluding the last)"
            )

    def apply_activation(values: Array, kind: str) -> Array:
        if kind == "tanh":
            return np.tanh(values)
        if kind == "relu":
            return np.maximum(values, 0)
        if kind == "identity":
            return values
        raise ValueError(f"unsupported activation '{kind}'")

    activations_iter = iter(activations)
    output = inputs
    for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
        output = output @ weight + bias
        is_last_layer = layer_idx == len(weights) - 1
        if not is_last_layer:
            activation = next(activations_iter, "tanh")
            output = apply_activation(output, activation)
    return output


__all__ = ["forward"]
