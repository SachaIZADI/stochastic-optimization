# -*- coding: utf-8 -*-

"""
Several implementations of the news vendor problem.
---------
In this problem, a paperboy has to buy N newspapers at cost C that he will sell the next day at price P.
The next-day demand for newspapers is random, and the paperboy needs to carefully build his inventory so
as to maximize his profits while hedging against loss.
"""

import math
from functools import cached_property
from logging import getLogger
from typing import List

import gurobipy as gp
import numpy as np
import scipy
from gurobipy import GRB
from scipy.stats import binom

logger = getLogger(__name__)


class Demand:

    EPS = 1e-10

    def __init__(self, rv: scipy.stats.rv_discrete, seed: int = 42) -> None:
        self.rv = rv
        self.rv.random_state = np.random.RandomState(seed=seed)

    @cached_property
    def values(self) -> List[int]:
        _min = self.rv.ppf(self.EPS)
        _max = self.rv.ppf(1 - self.EPS)
        return [*range(math.floor(_min), math.ceil(_max) + 1)]

    def samples(self, sample_size: int) -> np.ndarray:
        return self.rv.rvs(sample_size)


def max_expected_profit_analytic_solution(
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
) -> float:
    """
    Analytically computes the solution (number of orders) of the news vendor problem - with max E[profit] objective
    order* = F^(-1)[(p - c) / p]
    """
    return demand.rv.ppf((unit_sales_price - unit_cost) / unit_sales_price)


def max_expected_profit_solution(
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
) -> float:
    """
    Solves the news vendor problem using stochastic linear programming
    with max E[profits] objective: E[profits] = ∑ proba[Ω] * profits[Ω]
    """

    model = gp.Model("news_vendor_expectation")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand.values, lb=0, ub=max(demand.values), name="sales")

    model.addConstrs(order - sales[d] >= 0 for d in demand.values)
    model.addConstrs(sales[d] <= d for d in demand.values)

    model.setObjective(
        gp.quicksum(
            (sales[d] * unit_sales_price - order * unit_cost) * demand.rv.pmf(d)
            for d in demand.values
        ),
        GRB.MAXIMIZE,
    )

    model.optimize()

    return order.X


def min_conditional_value_at_risk_solution(
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
    alpha: float,
) -> float:
    """
    Solves the news vendor problem using stochastic linear programming
    with min CVaR_a[loss] objective.
    We use the following trick to compute CVaR: CVaR_a[loss[D]] = min t + E[|loss[D] - t|+] / (1 - a)
    """

    model = gp.Model("news_vendor_CVaR")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand.values, lb=0, ub=max(demand.values), name="sales")

    t = model.addVar(lb=-GRB.INFINITY, name="t")
    # excess := |loss[Ω] - t|+
    excess = model.addVars(demand.values, lb=0, name="excess")
    # profit := -loss
    profit = model.addVars(demand.values, lb=-GRB.INFINITY, name="profit")

    model.addConstrs(order - sales[d] >= 0 for d in demand.values)
    model.addConstrs(sales[d] <= d for d in demand.values)
    model.addConstrs(
        profit[d] == sales[d] * unit_sales_price - order * unit_cost
        for d in demand.values
    )

    model.addConstrs(excess[d] >= -profit[d] - t for d in demand.values)

    # fmt: off
    model.setObjective(
        t + gp.quicksum(
            (excess[d] / (1 - alpha)) * demand.rv.pmf(d) for d in demand.values
        ),
        GRB.MINIMIZE,
    )
    # fmt: on

    model.optimize()

    return order.X


def min_value_at_risk_solution(
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
    alpha: float,
) -> float:
    """
    Solves the news vendor problem using stochastic linear programming
    with min VaR_a[loss] objective.
    We need to introduce binary variables (therefore the problem becomes a MIP) in order
    to compute quantiles. The formulation is quite poor as it relies on large big-M tricks.
    This seems coherent with the remark from https://www.youtube.com/watch?v=Jb4a8T5qyVQ
    > "it becomes a "bad" MIP, need to track every-scenario"

    To compute the VaR we use the following trick:
    > VaR_a[L[D]] := min { v | P(L[D] ≤ v) ≥ a } (with L[D] being the loss, that depends on the demand r.v.)
    > We have P(L[D] ≤ v) = E[ indicator(L[D] ≤ v) ] = ∑_{d} P(D=d) * indicator(L[d] ≤ v)
    > The role of the introduced binary variables is to compute is_risk_lower_than_VaR[d] := indicator(L[d] ≤ VaR)
    """

    model = gp.Model("news_vendor_VaR")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand.values, lb=0, ub=max(demand.values), name="sales")
    value_at_risk = model.addVar(name="value_at_risk", lb=-1e6, ub=1e6)
    risk = model.addVars(demand.values, name="risk", lb=-1e6, ub=1e6)  # risk := loss
    # is_risk_lower_than_VaR[d] := 1 if value_at_risk >= risk[d], else 0
    is_risk_lower_than_VaR = model.addVars(
        demand.values, vtype=GRB.BINARY, name="is_risk_lower_than_VaR"
    )

    model.addConstrs(order - sales[d] >= 0 for d in demand.values)
    model.addConstrs(sales[d] <= d for d in demand.values)

    model.addConstrs(
        risk[d] == order * unit_cost - sales[d] * unit_sales_price
        for d in demand.values
    )

    # We use a big-M trick here to linearize the constraint `is_risk_lower_than_VaR[d] := 1 if value_at_risk >= risk[d], else 0`
    # L * (1 - is_risk_lower_than_VaR[d]) <= value_at_risk - risk[d] <= U * count_value_at_risk[d]
    # where L and U are lower and upper bounds for `value_at_risk - risk[d]`
    # TODO: remove hardcoded variables
    L, U = -1e6, +1e6
    model.addConstrs(
        L * (1 - is_risk_lower_than_VaR[d]) <= value_at_risk - risk[d]
        for d in demand.values
    )
    model.addConstrs(
        value_at_risk - risk[d] <= U * (is_risk_lower_than_VaR[d])
        for d in demand.values
    )

    # VaR definition as a 1-alpha quantile
    model.addConstr(
        gp.quicksum(is_risk_lower_than_VaR[d] * demand.rv.pmf(d) for d in demand.values)
        >= alpha
    )

    model.setObjective(value_at_risk, GRB.MINIMIZE)

    model.optimize()

    return order.X


if __name__ == "__main__":

    N = 10
    P = 0.5
    demand = Demand(scipy.stats.binom(N, P))

    unit_cost = 1
    unit_sales_price = 2

    print("============================")
    analytics_solution = max_expected_profit_analytic_solution(
        demand, unit_cost, unit_sales_price
    )
    print("analytics:")
    print(analytics_solution)

    print("============================")
    expectation_solution = max_expected_profit_solution(
        demand, unit_cost, unit_sales_price
    )
    print("max expected profits")
    print(expectation_solution)

    print("============================")
    for alpha in (0.99, 0.95, 0.85, 0.75, 0.5, 0.25, 0.15, 0.05, 0.01):
        try:
            cvar_solution = min_conditional_value_at_risk_solution(
                demand,
                unit_cost,
                unit_sales_price,
                alpha,
            )
            print(f"min CVaR {alpha}")
            print(cvar_solution)
        except:  # pylint: disable=bare-except
            ...

    print("============================")
    for alpha in (0.99, 0.95, 0.5, 0.25, 0.15, 0.05, 0.01):
        var_solution = min_value_at_risk_solution(
            demand,
            unit_cost,
            unit_sales_price,
            alpha,
        )
        print(f"min VaR {alpha}")
        print(var_solution)
