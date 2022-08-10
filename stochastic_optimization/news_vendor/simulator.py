# -*- coding: utf-8 -*-
from logging import getLogger
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from stochastic_optimization.news_vendor.news_vendor import Demand

logger = getLogger(__name__)


def simulate_profits(
    demand: Demand,
    unit_cost: float,
    unit_sales_price: float,
    order: float,
    sample_size: int = 1000,
) -> np.ndarray:
    """
    Simulates a news vendor profits for a given random demand, unit cost & price, and a chosen order
    """
    demand_samples = demand.samples(sample_size=sample_size)
    sales = np.minimum(order, demand_samples)
    profits = sales * unit_sales_price - order * unit_cost
    return profits


def plot_distribution(samples: Iterable[float], title: Optional[str] = None) -> None:
    """
    Helper function to plot the distribution and cumulative distribution of a series of samples
    from a random variable
    """

    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.hist(samples, density=True, label="Distribution", color="coral")
    ax2.hist(
        samples,
        density=True,
        histtype="step",
        cumulative=True,
        label="Cumulative distribution",
        color="darkred",
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    if title:
        plt.title(title)

    plt.show()


if __name__ == "__main__":

    N = 10
    P = 0.5
    sample_size = 10000

    demand = Demand(binom(N, P))
    samples = demand.samples(sample_size)
    plot_distribution(samples, "Demand")

    unit_cost = 1
    unit_sales_price = 2

    profits = simulate_profits(
        demand=demand,
        unit_cost=unit_cost,
        unit_sales_price=unit_sales_price,
        order=5,
        sample_size=sample_size,
    )
    plot_distribution(profits, "Profit")
