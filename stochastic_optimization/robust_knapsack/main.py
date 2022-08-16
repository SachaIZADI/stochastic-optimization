# -*- coding: utf-8 -*-

"""
Main entrypoint to the robust knapsack implementation. Use this script to run an instance of the problem,
and get a solution.
---------
In this problem, we are constrained by a fixed-size knapsack and must fill it with the most valuable items.
Each item has a cost that holds some uncertainty, price might vary depending on exogenous conditions.
"""
import itertools
from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

logger = getLogger(__name__)


@dataclass
class Item:
    value: float
    min_weight: float
    max_weight: float

    @property
    def delta_weight(self) -> float:
        return self.max_weight - self.min_weight


def solve_robust_knapsack(
    items: List[Item], capacity: float, uncertainty_budget: int = 1
) -> Tuple[List[int], float]:
    """
    Solves the robust knapsack problem. Returns a list of 1/0 corresponding to the  chosen items and the objective
    value at optimality.
    ---------------------------------
    The problem pseudo-formulation is the following:

        maximize knapsack objective under the worst scenario (with uncertainty budget)

    We use the formulation of https://xiongpengnus.github.io/rsome/example_ro_knapsack
        max ∑ value[i] * x[i]
        st. • ∑_{i} (min_weight[i] + z[i] * delta_weight[i]) * x[i] ≤ capacity  for all z in Z={|z|_0 = robust_budget}
            • [Not a constraint] : delta_weight[i] := max_weight[i] - min_weight[i]

    To represent the "for all z in Z={|z|_0 = robust_budget}" we need to introduce new constraints and explictly define
    the z's.
    NOTE: This is not the most elegant solution as it requires to pre-compute many scenarios ahead of time and introduce
    one constraint per scenario. There might be a smarter way of generating these scenarios on-the-fly (i.e. with a "good"
    meta-heuristic, column generation, etc.)

    New formulation:
        max ∑ value[i] * x[i]
        st. • ∑_{i} (min_weight[i] + Z[j, i] * delta_weight[i]) * x[i] ≤ capacity, for all j
            • Z = { [ z[j,0], z[j,1], ...] | ∑z[•, i] = robust_budget}
            • x binary
    """

    model = gp.Model("robust_knapsack")

    items_indexes = [*range(len(items))]
    Z = [
        [1 if i in combination else 0 for i in items_indexes]
        for combination in itertools.combinations(items_indexes, uncertainty_budget)
    ]
    Z_indexes = [*range(len(Z))]

    x = model.addVars(items_indexes, vtype=GRB.BINARY, name="x")

    model.addConstrs(
        gp.quicksum(
            (items[i].min_weight + Z[j][i] * items[i].delta_weight) * x[i]
            for i in items_indexes
        )
        <= capacity
        for j in Z_indexes
    )

    model.setObjective(
        gp.quicksum(x[i] * items[i].value for i in items_indexes), GRB.MAXIMIZE
    )

    model.optimize()

    return [x[i].X for i in items_indexes], model.ObjVal
