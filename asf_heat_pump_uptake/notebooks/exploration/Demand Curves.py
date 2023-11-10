# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Basically the idea is: on the price side, you have two groups, one of which has a low discount rate and one a high discount rate. The low discount rate people care more about running costs
#
# Then there is a demand curve with 2 (or as many as we like) groups, who have a simple (currently made up) relationship between price and likelihood of choosing a heat pump when they switch. If you plug a given price in (and know the size of each group), you get an estimate of uptake.
#
# This is all quite simple, and we can add things (like declining HP prices), but I think it gives us something we can play some different tunes on (and let people vary)
#
# https://docs.google.com/spreadsheets/d/1wGPFrQI4-ikL-4YSR572YcqDriY4J2XxqDjqT48ibn0/edit?usp=sharing

# %%
import numpy
from dataclasses import dataclass
from typing import Sequence
import matplotlib.pyplot as plt


# %%
@dataclass
class HeatSource:
    """Class for heat sources."""

    name: str
    total_upfront_cost: float
    lifespan: float
    heat_demand: float  # kWh
    efficiency: float
    fuel_price: float  # in pence per kWh

    @property
    def annual_upfront_cost(self) -> float:
        return self.total_upfront_cost / self.lifespan

    @property
    def annual_running_cost(self) -> float:
        return (self.heat_demand / self.efficiency) * (self.fuel_price / 100)


# %%
@dataclass
class Population:
    name: str
    size: float


@dataclass
class SubPopulation(Population):
    group_name: str
    group_share: float
    discount_rate: float

    @property
    def group_size(self) -> float:
        return self.size * self.group_share


# %%
# NB why aren't we counting the zeroth year?
def calculate_running_costs_net_present_value(
    group: SubPopulation, heat_source: HeatSource
) -> float:
    """Calculate the net present value of running costs."""
    return sum(
        [
            heat_source.annual_running_cost / (1 + group.discount_rate) ** t
            for t in range(1, 21)
        ]
    )


def calculate_annualised_running_costs_net_present_value(
    group: SubPopulation, heat_source: HeatSource
) -> float:
    """Calculate the annualised net present value of running costs."""
    return calculate_running_costs_net_present_value(group, heat_source) / 20


def calculate_total_annualised_net_present_value(
    group: SubPopulation, heat_source: HeatSource
) -> float:
    """Calculate the annualised net present value of running costs."""
    return (
        calculate_annualised_running_costs_net_present_value(group, heat_source)
        + heat_source.annual_upfront_cost
    )


# %%
# Fossil Fuel Heating (e.g. Gas Boiler)
gas_boiler = HeatSource("Gas Boiler", 2500, 15, 12000, 0.85, 5)
# Low carbon heating (e.g. heat pump)
heat_pump = HeatSource("Heat Pump", 12000, 20, 12000, 3, 15)

# %%
group_1 = SubPopulation("Households", 29_000_000, "Group 1", 0.3, 0.03)
group_2 = SubPopulation("Households", 29_000_000, "Group 2", 0.7, 0.06)

# %%
# Demand Curve
relative_price = numpy.arange(-1000, 1050, 50)[::-1]
group_A_quantity = [
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.22,
    0.24,
    0.26,
    0.28,
    0.3,
    0.32,
    0.34,
    0.36,
    0.38,
    0.4,
    0.42,
    0.44,
    0.46,
    0.48,
    0.5,
    0.52,
    0.54,
    0.56,
    0.58,
    0.6,
    0.62,
    0.64,
    0.66,
    0.68,
    0.7,
    0.72,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
group_B_quantity = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.16,
    0.24,
    0.32,
    0.4,
    0.48,
    0.56,
    0.64,
    0.72,
    0.8,
    0.88,
    0.96,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]

# %%
f, ax = plt.subplots(figsize=(8, 5))

ax.plot(group_A_quantity, relative_price, label="Group A Demand")
ax.plot(group_B_quantity, relative_price, label="Group B Demand")
ax.grid()
ax.legend()
ax.set_xlabel("Normalised Quantity")
ax.set_ylabel("Price of heat pump relative to gas boiler")


# %%
def get_normalised_quantity_for_relative_price(
    candidate_value: float,
    relative_price_distribution: Sequence,
    group_quantity: Sequence,
) -> float:
    """Get the group quantity for an arbitrary value against the deamnd curve."""
    if candidate_value in relative_price_distribution:
        # Special case of exact match for candidate price.
        index = list(relative_price_distribution).index(candidate_value)
        return group_quantity[index]
    else:
        # Linear interpolation when value between two values.
        index = len(relative_price_distribution) - numpy.searchsorted(
            relative_price_distribution[::-1], candidate_value, side="left"
        )
        lower, upper = (
            relative_price_distribution[index],
            relative_price_distribution[index - 1],
        )
        interpolation_factor = (candidate_value - lower) / (upper - lower)
        return abs(
            ((group_quantity[index] - group_quantity[index - 1]) * interpolation_factor)
            - group_quantity[index]
        )


# %%
# calculate relative price for group 1
relative_price_group_1 = calculate_total_annualised_net_present_value(
    group_1, heat_pump
) - calculate_total_annualised_net_present_value(group_1, gas_boiler)
relative_price_group_1

# %%
# calculate relative price for group 1
relative_price_group_2 = calculate_total_annualised_net_present_value(
    group_2, heat_pump
) - calculate_total_annualised_net_present_value(group_2, gas_boiler)
relative_price_group_2

# %%
# Group 1: Now get normalised quantities for group A and group B
quantity_group_1_group_A = get_normalised_quantity_for_relative_price(
    relative_price_group_1, relative_price, group_A_quantity
)
quantity_group_1_group_B = get_normalised_quantity_for_relative_price(
    relative_price_group_1, relative_price, group_B_quantity
)
quantity_group_1_group_A, quantity_group_1_group_B

# %%
# Get Group 1 total conversions
group_1_total = (quantity_group_1_group_A * group_1.group_size) + (
    quantity_group_1_group_B * group_1.group_size
)

# %%
# Group 2: Now get normalised quantities for group A and group B
quantity_group_2_group_A = get_normalised_quantity_for_relative_price(
    relative_price_group_2, relative_price, group_A_quantity
)
quantity_group_2_group_B = get_normalised_quantity_for_relative_price(
    relative_price_group_2, relative_price, group_B_quantity
)
quantity_group_2_group_A, quantity_group_2_group_B

# %%
# Get Group 2 total conversions
group_2_total = (quantity_group_2_group_A * group_2.group_size) + (
    quantity_group_2_group_B * group_2.group_size
)

# %%
# Total
group_1_total + group_2_total

# %%
# Proportion
(group_1_total + group_2_total) / (group_1.group_size + group_2.group_size)

# %% [markdown]
# ## Development
#
# * Identify when gas boilers reach the end of their life. (Gas Safe data).
# * Specify decision mechanism for end of life boilers (e.g. amount to replace at given time).
# * Identify key groupings for demand curves (Group A & B: environmentally motivated & cost motivated... others?).
# * Specify demand curve at $t_{0}$
# * Specify demand curves for t > 0.
# * Identify key groupings for discount rates (Group 1 & 2: Present focused & future focused).
# * Specify discount rates at $t_{0}$.
# * Specify discount rates for t > 0.
# * Specify upfront cost of gas boiler & heat pump at for all t. (mean or distribution to account for housing stock).
# * Specify running cost (fuel cost of gas and electricity) for all t. (nb cnz abm assumes these are fixed for time of the simulation.)
# * Consider supply side.
#
# ## Interventions
#
# * Loan mechanism for upfront costs (e.g. private loans, green loans, government-backed etc.)
# * Operation of a boiler ban (e.g. effect of announcement; Actual ban).
# * Market mechanism effect on unit prices or gas boilers and heat pumps?
#
# ## Technical
#
# * Implement simulation as micro (e.g. individual) or macro (e.g. aggregate) simulation?
# * Implement time.
# * Implement tracking of variables.
# * Decide on variable manipulation and implement basic dashboarding.

# %% [markdown]
# ## Behaviour
#
# * Discount rates - compatible with behaviour lit. viz hyperbolic discounting (behavioural science research)
#  - The closer you get to an event, the discount rate varies by proximity to event (see pic)
#  - strength of status quo bias.
#  - status quo bias a degree of correaltion with - past (traditional), present bias, future focused.
#
# Talk to Toby in sustainability in BIT.
#
# Or interaction between deadline effect and status quo bias.
#
# What is the most likely glide path? - animation of build.
#
# Logic to build in for text that describes what is going on.
#
# Elspeth - distilling back to my original thought that spurred this, it's def about the mad shit people do in the final years before the ban, so status quo bias (I like what I know) + deadline effect (I will leave it to the last min to decide)

# %% [markdown]
# ![image.png](attachment:image.png)
