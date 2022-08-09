# -*- coding: utf-8 -*-
import math
from functools import cached_property
from logging import getLogger
from typing import List

import gurobipy as gp
import matplotlib.pyplot as plt
import scipy
from gurobipy import GRB
from scipy.stats import binom

logger = getLogger(__name__)

"""
Several implementations of the news vendor problem.
---------
In this problem, a paperboy has to buy N newspapers at cost C that he will sell the next day at price P.
The next-day demand for newspapers is random, and the paperboy needs to carefully build his inventory so
as to maximize his profits while hedging against loss
"""


class Demand:

    EPS = 1e-10

    def __init__(self, rv: scipy.stats.rv_discrete) -> None:
        self.rv = rv

    @cached_property
    def values(self) -> List[int]:
        _min = self.rv.ppf(self.EPS)
        _max = self.rv.ppf(1 - self.EPS)
        return [*range(math.floor(_min), math.ceil(_max) + 1)]


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
    We use the following trick to compute CVaR: CVaR_a[loss[Ω]] = min t + E[|loss[Ω] - t|+] / (1 - a)
    """

    #FIXME: it doesn't seem to solve for all values of alpha. To be investigated…

    model = gp.Model("news_vendor_CVaR")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand.values, lb=0, ub=max(demand.values), name="sales")

    t = model.addVar(lb=-GRB.INFINITY, name="t")
    excess = model.addVars(demand.values, lb=0, name="excess")  # excess := |loss[Ω] - t|+
    profit = model.addVars(demand.values, name="profit", lb=-GRB.INFINITY)  # profit := -loss

    model.addConstrs(order - sales[d] >= 0 for d in demand.values)
    model.addConstrs(sales[d] <= d for d in demand.values)
    model.addConstrs(
        profit[d] == sales[d] * unit_sales_price - order * unit_cost
        for d in demand.values
    )

    model.addConstrs(excess[d] >= -profit[d] - t for d in demand.values)

    model.setObjective(
        gp.quicksum(t + (excess[d] / (1 - alpha)) * demand.rv.pmf(d) for d in demand.values),
        GRB.MINIMIZE,
    )

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
    """

    #FIXME: it doesn't seem to solve for all values of alpha. To be investigated…

    model = gp.Model("news_vendor_VaR")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand.values, lb=0, ub=max(demand.values), name="sales")
    value_at_risk = model.addVar(name="value_at_risk", lb=-1e6, ub=1e6)
    risk = model.addVars(demand.values, name="risk", lb=-1e6, ub=1e6)  # risk := loss
    # bin_value_at_risk[d] := 1 if value_at_risk >= risk[d], else 0
    bin_value_at_risk = model.addVars(
        demand.values, vtype=GRB.BINARY, name="bin_value_at_risk"
    )


    model.addConstrs(order - sales[d] >= 0 for d in demand.values)
    model.addConstrs(sales[d] <= d for d in demand.values)

    model.addConstrs(
        risk[d] == order * unit_cost - sales[d] * unit_sales_price
        for d in demand.values
    )

    # We use a big-M trick here to linearize the constraint `bin_value_at_risk[d] := 1 if value_at_risk >= risk[d], else 0`
    # L * (1 - bin_value_at_risk[d]) <= value_at_risk - risk[d] <= U * count_value_at_risk[d]
    # where L and U are lower and upper bounds for `value_at_risk - risk[d]`
    # TODO: remove hardcoded variables
    L, U = -1e6, +1e6
    model.addConstrs(
        L * (1 - bin_value_at_risk[d]) <= value_at_risk - risk[d]
        for d in demand.values
    )
    model.addConstrs(
        value_at_risk - risk[d] <= U * (bin_value_at_risk[d]) for d in demand.values
    )

    # We have VaR_a[X] := min { v | P(X ≤ v) = 1 - a } (with X being the value at risk - or the loss here)
    # Some maths here
    # TODO: write the logic
    # P(X ≤ v) = ∑_{x ≤ v} P(X=x) 
    # = E[ indicator{X ≤ v} ] =
    # To introduce this constraint we have 
    model.addConstr(
        gp.quicksum(bin_value_at_risk[d] * demand.rv.pmf(d) for d in demand.values) <= 1 - alpha
    )
    # TODO: remove the hardcoded slack variable -0.1
    model.addConstr(
        gp.quicksum(bin_value_at_risk[d] * demand.rv.pmf(d) for d in demand.values)
        >= 1 - alpha - 0.1 
    )

    model.setObjective(value_at_risk, GRB.MINIMIZE)

    model.optimize()

    return order.X


def main() -> None:

    N = 500
    P = 0.5
    SAMPLE_SIZE = 100
    PLOT = True

    demand = binom(N, P)
    demand_values = [*range(N + 1)]
    probabilities = [(d, demand.pmf(d)) for d in demand_values]
    samples = demand.rvs(SAMPLE_SIZE)

    if PLOT:
        plt.hist(samples, SAMPLE_SIZE, density=True)
        plt.hist(
            samples,
            SAMPLE_SIZE,
            density=True,
            histtype="step",
            cumulative=True,
            label="Empirical",
        )
        plt.plot([p[0] for p in probabilities], [p[1] for p in probabilities])
        plt.show()


def main_ter():
    N = 10
    P = 0.5
    SAMPLE_SIZE = 100
    PLOT = False

    demand = binom(N, P)
    demand_values = [*range(N + 1)]
    probabilities = [(d, demand.pmf(d)) for d in demand_values]
    samples = demand.rvs(SAMPLE_SIZE)

    unit_cost = 1
    unit_sales_price = 2

    model = gp.Model("news_vendor")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand_values, lb=0, ub=max(demand_values), name="sales")
    value_at_risk = model.addVar(name="value_at_risk", lb=-1e6, ub=1e6)
    risk = model.addVars(demand_values, name="risk", lb=-1e6, ub=1e6)
    # count_value_at_risk[d] = 1 if value_at_risk >= risk[d] , else 0
    count_value_at_risk = model.addVars(
        demand_values, vtype=GRB.BINARY, name="count_value_at_risk"
    )

    # Minimize VaR-75%
    p = 0.15

    model.addConstrs(order - sales[d] >= 0 for d in demand_values)
    model.addConstrs(sales[d] <= d for d in demand_values)

    model.addConstrs(
        risk[d] == order * unit_cost - sales[d] * unit_sales_price
        for d in demand_values
    )

    L = -1e6
    U = +1e6
    # count_value_at_risk[d] = 1 if value_at_risk >= risk[d] , else 0
    # b = 1 ==> value_at_risk - risk[d] ≤ 0 ==> value_at_risk ≤ risk[d]
    model.addConstrs(
        L * (1 - count_value_at_risk[d]) <= value_at_risk - risk[d]
        for d in demand_values
    )
    model.addConstrs(
        value_at_risk - risk[d] <= U * (count_value_at_risk[d]) for d in demand_values
    )
    # model.addConstrs(
    #     count_value_at_risk[d] >= count_value_at_risk[d + 1] for d in [*range(N)]
    # )

    model.addConstr(
        gp.quicksum(count_value_at_risk[d] * demand.pmf(d) for d in demand_values) <= p
    )

    model.addConstr(
        gp.quicksum(count_value_at_risk[d] * demand.pmf(d) for d in demand_values)
        >= p - 0.1
    )

    model.setObjective(
        value_at_risk,
        GRB.MINIMIZE,
    )
    model.write("test.lp")

    # Optimize model
    model.optimize()

    for v in model.getVars():
        print("%s %g" % (v.VarName, v.X))

    print("Obj: %g" % model.ObjVal)


if __name__ == "__main__":
    # main()
    # main_bis()
    # main_ter()

    N = 10
    P = 0.5
    demand = Demand(scipy.stats.binom(N, P))

    unit_cost = 1
    unit_sales_price = 2

    print("============================")
    analytics_solution = max_expected_profit_analytic_solution(
        demand, unit_cost, unit_sales_price
    )
    print(analytics_solution)

    print("============================")
    expectation_solution = max_expected_profit_solution(
        demand, unit_cost, unit_sales_price
    )
    print(expectation_solution)

    for alpha in (0.5, 0.75, 0.85, 0.95):
        try:
            print("============================")
            cvar_solution = min_conditional_value_at_risk_solution(
                demand, unit_cost, unit_sales_price, alpha
            )
            print(cvar_solution)
        except:
            ...
        
