# -*- coding: utf-8 -*-

"""
Main entrypoint to the robust knapsack implementation. Use this script to run an instance of the problem,
and get a solution.
---------
In this problem, we are constrained by a fixed-size knapsack and must fill it with the most valuable items.
Each item has a cost that holds some uncertainty, price might vary depending on exogenous conditions.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import List

import gurobipy as gp
from gurobipy import GRB

logger = getLogger(__name__)


@dataclass
class Item:
    value: float
    min_price: float
    max_price: float


def solve_robust_knapsack(items: List[Item], uncertainty_budget: int = 1) -> List[int]:
    """
    Solves the robust knapsack problem. Returns a list of integers corresponding
    to the items indexes chosen.
    """
    # TODO: to be implemented
    print(gp, GRB, items, uncertainty_budget)
    ...


if __name__ == "__main__":
    solve_robust_knapsack(items=[], uncertainty_budget=2)
