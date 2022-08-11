# -*- coding: utf-8 -*-
from logging import getLogger
from typing import Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stochastic_optimization.news_vendor.optimizer import Demand

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


def plot_distribution(
    samples: Iterable[float],
    title: Optional[str] = None,
    outstanding_points: Iterable[Tuple[str, float]] = (),
) -> None:
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

    if title:
        plt.title(title)

    colors = matplotlib.cm.cool(np.linspace(0, 1, len(outstanding_points)))
    for i, outstanding_point in enumerate(outstanding_points):
        ax2.vlines(
            x=outstanding_point[1],
            ymin=0,
            ymax=1,
            label=outstanding_point[0],
            color=colors[i],
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=2)

    plt.show()
