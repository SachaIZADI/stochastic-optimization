# -*- coding: utf-8 -*-
"""
Main entrypoint to the news vendor problem implementation. Use this script to run an instance of the problem,
get a solution and plot the profits distribution to visualize if you're hedging against risk.
"""
from enum import Enum
from typing import List, Optional

import numpy as np
import scipy

from stochastic_optimization.news_vendor.optimizer import (
    Demand,
    max_expected_profit_analytic_solution,
    max_expected_profit_solution,
    min_conditional_value_at_risk_solution,
    min_value_at_risk_solution,
)
from stochastic_optimization.news_vendor.simulator import (
    plot_distribution,
    simulate_profits,
)


def get_scipy_discrete_distributions() -> List[str]:
    """Returns the list of scipy discrete distributions supported by the model"""
    discrete_distributions: List[str] = []
    for distribution_name in dir(scipy.stats):
        if isinstance(getattr(scipy.stats, distribution_name), scipy.stats.rv_discrete):
            discrete_distributions.append(distribution_name)
    return discrete_distributions


class ProblemInstance(Enum):
    expected_profit_analytic = "expected_profit_analytic"
    expected_profit_lp = "expected_profit_lp"
    VaR = "VaR"
    CVaR = "CVaR"

    @property
    def solve(self) -> "function":
        if self == self.expected_profit_analytic:
            return max_expected_profit_analytic_solution

        if self == self.expected_profit_lp:
            return max_expected_profit_solution

        if self == self.VaR:
            return min_value_at_risk_solution

        if self == self.CVaR:
            return min_conditional_value_at_risk_solution

        raise NotImplementedError()

    @property
    def is_alpha_expected(self) -> bool:
        return self in [self.VaR, self.CVaR]


def solve(
    problem: ProblemInstance,
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
    alpha: Optional[float] = None,
    sample_size: int = 1000,
) -> None:

    plot_distribution(
        demand.samples(sample_size),
        title="Demand",
        outstanding_points=[("Average demand", demand.rv.mean())],
    )

    kwargs = {}
    if problem.is_alpha_expected:
        kwargs["alpha"] = alpha if alpha is not None else 0.75

    order = problem.solve(demand, unit_cost, unit_sales_price, **kwargs)

    profits = simulate_profits(
        demand,
        unit_cost,
        unit_sales_price,
        order,
        sample_size,
    )

    plot_distribution(
        profits,
        title=f"Profits - {problem.name}",
        outstanding_points=[
            ("Null profit", 0),
            ("Expected profit", np.mean(profits)),
            ("Min profit", np.min(profits)),
        ],
    )
