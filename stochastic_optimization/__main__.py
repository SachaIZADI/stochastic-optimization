# -*- coding: utf-8 -*-
from typing import Dict

import click
import scipy

from stochastic_optimization.news_vendor.main import (
    ProblemInstance,
    get_scipy_discrete_distributions,
    solve,
)
from stochastic_optimization.news_vendor.optimizer import Demand
from stochastic_optimization.robust_knapsack.main import solve_robust_knapsack


@click.group()
def cli() -> None:
    ...


@cli.command(
    name="news-vendor",
    short_help="Runs the news-vendor problem",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--problem-instance",
    type=click.Choice(ProblemInstance.__members__),
    default=ProblemInstance.expected_profit_analytic.value,
    help="NewsVendor problem instance to solve",
)
@click.option(
    "--demand-distribution",
    type=click.Choice(get_scipy_discrete_distributions()),
    default="binom",
    help=(
        "scipy.stats distribution to chose for the demand - please make sure "
        "it has a limited spread of values (to limit instance size). "
        "You can pass distribution parameters as additional (unchecked) options "
        "- see README for examples"
    ),
)
@click.option(
    "--unit-cost",
    type=float,
    default=1.0,
    help="unit cost of a newspaper",
)
@click.option(
    "--unit-sales-price",
    type=float,
    default=2.0,
    help="unit sales price of a newspaper",
)
@click.option(
    "--alpha",
    type=float,
    help="Percentile (between 0.0 and 1.0) used for VaR and CVaR objectives",
)
@click.option(
    "--sample-size",
    type=int,
    help="Sample size for plotting the distribution graphs",
    default=10_000,
)
@click.pass_context
def cli_news_vendor(
    ctx: click.Context,
    problem_instance: ProblemInstance,
    demand_distribution: str,
    unit_cost: float,
    unit_sales_price: float,
    alpha: float,
    sample_size: int,
) -> None:

    # We use click context to pass ad-hoc arguments to parametrize the demand distribution
    # Credit to: https://stackoverflow.com/questions/32944131/add-unspecified-options-to-cli-command-using-python-click
    distribution_kwargs: Dict[str, float] = {
        ctx.args[i][2:]: float(ctx.args[i + 1])
        if not float(ctx.args[i + 1]).is_integer()
        else int(ctx.args[i + 1])
        for i in range(0, len(ctx.args), 2)
    }
    # Just a small check to handle coherent default values
    if not distribution_kwargs and demand_distribution == "binom":
        distribution_kwargs = {"n": 10, "p": 0.5}

    demand_rv = getattr(scipy.stats, demand_distribution)(**distribution_kwargs)
    solve(
        problem=ProblemInstance(problem_instance),
        demand=Demand(demand_rv),
        unit_cost=unit_cost,
        unit_sales_price=unit_sales_price,
        alpha=alpha,
        sample_size=sample_size,
    )


@cli.command(
    name="robust-knapsack",
    short_help="Runs the robust knapsack problem",
)
@click.option(
    "--uncertainty-budget",
    type=int,
    default=1,
    help="Uncertainty budget to robust resolution",
)
def cli_robust_knapsack(uncertainty_budget: int) -> None:
    # TODO: add parameter to define items
    solve_robust_knapsack(items=[], uncertainty_budget=uncertainty_budget)


if __name__ == "__main__":
    cli()
