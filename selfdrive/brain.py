"""
The neural network ("brain") of a car: a tiny fully-connected network.

    inputs  : distance-sensor readings (0 = wall touching, 1 = nothing in range)
              + current speed (0..1)
    hidden  : tanh layers
    outputs : steer (-1 left .. +1 right) and throttle (-1 brake .. +1 gas)

A "genome" is just all weights and biases flattened into one vector. The whole
population is a (P, G) matrix and is evaluated in one batched forward pass, so
100 cars cost about the same as 1.
"""
from __future__ import annotations

import numpy as np


class Brain:
    def __init__(self, layers: list[int]):
        self.layers = list(layers)
        self.shapes = [(a, b) for a, b in zip(layers[:-1], layers[1:])]
        self.size = sum(a * b + b for a, b in self.shapes)

    def unpack(self, genomes: np.ndarray):
        """(P, G) genome matrix -> list of (W (P,in,out), b (P,out)) per layer."""
        P = genomes.shape[0]
        params, k = [], 0
        for a, b in self.shapes:
            W = genomes[:, k:k + a * b].reshape(P, a, b)
            k += a * b
            bias = genomes[:, k:k + b]
            k += b
            params.append((W, bias))
        return params

    @staticmethod
    def forward(params, x: np.ndarray) -> np.ndarray:
        """x: (P, n_inputs) -> (P, n_outputs) in [-1, 1]. Car i uses network i."""
        h = x
        for W, b in params:
            h = np.tanh(np.einsum("pi,pio->po", h, W) + b)
        return h

    def init_population(self, P: int, rng: np.random.Generator) -> np.ndarray:
        # Scale each layer's weights by 1/sqrt(fan_in) (Xavier-like) so the
        # random initial cars produce varied but non-saturated outputs.
        cols = []
        for a, b in self.shapes:
            cols.append(rng.normal(0, 1 / np.sqrt(a), (P, a * b)))
            cols.append(rng.normal(0, 0.1, (P, b)))
        return np.concatenate(cols, axis=1)
