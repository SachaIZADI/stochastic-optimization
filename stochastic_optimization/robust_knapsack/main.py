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


def solve_robust_knapsack(
    items: List[Item], capacity: float, uncertainty_budget: int = 1
) -> List[int]:
    """
    Solves the robust knapsack problem. Returns a list of integers corresponding
    to the items indexes chosen.
    ---------------------------------
    The problem formulation is the following:

        maximize knapsack objective under the worst scenario (with uncertainty budget)

        # FIXME: this is not the right formulation ...

        max_{item choices} min_{worst case scenarios} (knapsack objective)
        st. • ∑ worst case scenarios == uncertainty budget
            • capacity constraint

    We rewrite the max-min problem as

       max t
       s.t. • t ≤ knapsack objective for all item choices (for a given worst case scenario)
            • ∑ worst case scenarios == uncertainty budget
            • capacity constraint

    """

    model = gp.Model("robust_knapsack")

    is_worst_price = model.addVars(
        [*range(len(items))], vtype=GRB.BINARY, name="is_worst_price"
    )
    is_item_selected = model.addVars(
        [*range(len(items))], vtype=GRB.BINARY, name="is_item_selected"
    )
    t = model.addVar(lb=-GRB.INFINITY, name="t")
    z = model.addVars([*range(len(items))], vtype=GRB.BINARY, name="z")

    # ∑ worst case scenarios == uncertainty budget
    model.addConstr(gp.quicksum(is_worst_price) == uncertainty_budget)

    # Capacity constraint
    #   We initially wrote it as:
    #      ∑ is_item_selected * (is_worst_price * item.max_price + (1 - is_worst_price) * item.min_price ≤ capacity
    #   Which turns into:
    #       ∑ is_item_selected * is_worst_price * (item.max_price - item.min_price) + is_item_selected * item.min_price ≤ capacity
    #   To linearize the constraint we introduce
    #       z = is_item_selected * is_worst_price

    model.addConstrs(z[i] <= is_item_selected[i] for i in range(len(items)))
    model.addConstrs(z[i] <= is_worst_price[i] for i in range(len(items)))
    model.addConstrs(
        z[i] >= is_item_selected[i] + is_worst_price[i] - 1 for i in range(len(items))
    )
    model.addConstr(
        gp.quicksum(
            z[i] * (items[i].max_price - items[i].min_price)
            + is_item_selected[i] * items[i].min_price
            for i in range(len(items))
        )
        <= capacity
    )

    # t ≥ knapsack objective
    model.addConstr(
        t
        <= gp.quicksum(is_item_selected[i] * items[i].value for i in range(len(items)))
    )

    model.setObjective(t, GRB.MAXIMIZE)

    model.optimize()

    for v in model.getVars():
        print("%s %g" % (v.VarName, v.X))

    print("Obj: %g" % model.ObjVal)


if __name__ == "__main__":

    items = [
        Item(value=12, min_price=3, max_price=7),
        Item(value=6, min_price=2, max_price=3),
        Item(value=5, min_price=2, max_price=3),
        Item(value=5, min_price=1, max_price=2),
    ]

    solve_robust_knapsack(items=items, uncertainty_budget=2, capacity=7)
