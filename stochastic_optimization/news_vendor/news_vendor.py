# -*- coding: utf-8 -*-
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB
from scipy.stats import binom


def main() -> None:

    N = 500
    P = 0.5
    SAMPLE_SIZE = 100
    PLOT = True

    demand = binom(N, P)
    demand_values = [*range(N + 1)]
    probabilities = [(d, demand.pmf(d)) for d in demand_values]
    samples = demand.rvs(SAMPLE_SIZE)

    unit_cost = 1
    unit_sales_price = 2

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

    model = gp.Model("news_vendor")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand_values, lb=0, ub=max(demand_values), name="sales")

    # Maximize expected profit
    model.setObjective(
        gp.quicksum(
            (sales[d] * unit_sales_price - order * unit_cost) * demand.pmf(d)
            for d in demand_values
        ),
        GRB.MAXIMIZE,
    )

    # Minimize CVaR-75%
    # CVaR_a[Ω] = min t + E[|Ω-t|+] / (1 - a)

    model.addConstrs(order - sales[d] >= 0 for d in demand_values)
    model.addConstrs(sales[d] <= d for d in demand_values)

    # Optimize model
    model.optimize()

    for v in model.getVars():
        print("%s %g" % (v.VarName, v.X))

    print("Obj: %g" % model.ObjVal)


def main_bis() -> None:

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

    model = gp.Model("news_vendor")

    order = model.addVar(lb=0, name="order")
    sales = model.addVars(demand_values, lb=0, ub=max(demand_values), name="sales")

    # Minimize CVaR-75%
    # CVaR_a[Ω] = min t + E[|-Ω-t|+] / (1 - a)
    a = 0.95

    t = model.addVar(lb=-1e6, name="t")
    excess = model.addVars(demand_values, lb=0, name="excess")
    profit = model.addVars(demand_values, name="profit", lb=-1e6)

    model.addConstrs(order - sales[d] >= 0 for d in demand_values)
    model.addConstrs(sales[d] <= d for d in demand_values)
    model.addConstrs(
        profit[d] == sales[d] * unit_sales_price - order * unit_cost
        for d in demand_values
    )

    model.addConstrs(excess[d] >= -profit[d] - t for d in demand_values)

    model.setObjective(
        gp.quicksum(t + (excess[d] / (1 - a)) * demand.pmf(d) for d in demand_values),
        GRB.MINIMIZE,
    )

    # Optimize model
    model.optimize()

    for v in model.getVars():
        print("%s %g" % (v.VarName, v.X))

    print("Obj: %g" % model.ObjVal)


if __name__ == "__main__":
    # main()
    main_bis()
