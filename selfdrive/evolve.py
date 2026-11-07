"""
Genetic algorithm (neuroevolution), same idea as the deepCars reference:

  - evaluate every car (genome) -> fitness
  - keep the best few unchanged       (elitism)
  - fill the rest with children:
        pick 2 parents by tournament  (better cars are picked more often)
        uniform crossover             (each weight from parent A or B)
        gaussian mutation             (randomly nudge some weights)

No gradients / backprop are needed: the "learning" is just selection pressure.
"""
from __future__ import annotations

import numpy as np


class GeneticAlgorithm:
    def __init__(self, pop_size: int = 100, elite: int = 4, tournament: int = 5,
                 crossover_rate: float = 0.7, mutation_rate: float = 0.1,
                 mutation_sigma: float = 0.3, rng: np.random.Generator | None = None):
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.rng = rng or np.random.default_rng()

    def _pick(self, fitness: np.ndarray) -> int:
        cand = self.rng.integers(len(fitness), size=self.tournament)
        return int(cand[np.argmax(fitness[cand])])

    def next_generation(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        P, G = pop.shape
        order = np.argsort(-fitness)
        children = [pop[order[: self.elite]]]
        n = self.pop_size - self.elite
        kids = np.empty((n, G))
        for i in range(n):
            a = pop[self._pick(fitness)]
            if self.rng.random() < self.crossover_rate:
                b = pop[self._pick(fitness)]
                kids[i] = np.where(self.rng.random(G) < 0.5, a, b)
            else:
                kids[i] = a
        mask = self.rng.random(kids.shape) < self.mutation_rate
        kids = kids + mask * self.rng.normal(0, self.mutation_sigma, kids.shape)
        children.append(kids)
        return np.concatenate(children, 0)
