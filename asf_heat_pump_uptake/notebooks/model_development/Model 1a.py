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

# %%
import numpy
from dataclasses import dataclass
from typing import Sequence
from copy import deepcopy
from matplotlib import pyplot
from scipy.stats import gaussian_kde


# %% [markdown]
# ## Model 1a
#
# NB the economics NPV functions in the Household class allow a given household to evaluate different heat sources. We assume that total upfront cost is a function of household and heat source, and hence is not a property of either those classes, although it could be decomposed into heat source unit costs, heating system (household costs) and (external) labour costs.

# %%
# Define some base classes


@dataclass
class Fuel:
    """Class for fuels."""

    name: str
    price: float  # unit cost in pence


@dataclass
class HeatSource:
    """Class for heat sources."""

    name: str
    lifespan: float
    efficiency: float
    fuel: Fuel


@dataclass
class DemandCurve:
    """Class for demand curves."""

    relative_price: Sequence
    probability: Sequence


@dataclass
class Household:
    """Class for households."""

    heat_demand: float  # kWh
    discount_rate: float
    demand_curve: DemandCurve
    heat_source: HeatSource

    def annual_upfront_cost(self, total_upfront_cost, heat_source) -> float:
        """Given total upfront cost and heat source, return annualised upfront cost for household."""
        return total_upfront_cost / heat_source.lifespan

    def annual_running_cost(self, heat_source) -> float:
        """Given heat source, return annual running cost for household."""
        return (self.heat_demand / heat_source.efficiency) * (
            heat_source.fuel.price / 100
        )

    def running_cost_net_present_value(self, heat_source) -> float:
        """Given heat source, return net present value of running costs."""
        return sum(
            [
                self.annual_running_cost(heat_source) / (1 + self.discount_rate) ** t
                for t in range(1, 21)
            ]
        )

    def annualised_running_costs_net_present_value(self, heat_source) -> float:
        """Given heat source, return annualised net present value of running costs."""
        return self.running_cost_net_present_value(heat_source) / 20

    def total_annualised_net_present_value(
        self, total_upfront_cost, heat_source
    ) -> float:
        """Given total upfront cost and heat source, return the total annualised net present."""
        return self.annualised_running_costs_net_present_value(
            heat_source
        ) + self.annual_upfront_cost(total_upfront_cost, heat_source)


# %%
# Define some model components from base classes

# Fuels
natural_gas = Fuel(name="Natural Gas", price=5)
electricity = Fuel(name="Electricity", price=15)

# Heat Sources
gas_boiler = HeatSource(
    name="Gas Boiler", lifespan=15, efficiency=0.85, fuel=natural_gas
)

heat_pump = HeatSource(name="Heat Pump", lifespan=20, efficiency=3.0, fuel=electricity)

# Demand Curve
demand_curve = DemandCurve(
    relative_price=[
        964,
        956,
        947,
        935,
        922,
        905,
        885,
        862,
        834,
        801,
        762,
        716,
        664,
        604,
        537,
        462,
        380,
        291,
        197,
        100,
        0.0,
        -100,
        -197,
        -291,
        -380,
        -462,
        -537,
        -604,
        -664,
        -716,
        -762,
        -801,
        -834,
        -862,
        -885,
        -905,
        -922,
        -935,
        -947,
        -956,
        -964,
        -970,
        -976,
        -980,
        -984,
        -987,
        -989,
        -991,
        -993,
        -994,
        -995,
    ],
    probability=numpy.linspace(0, 1, 51),
)

# %%
f, ax = pyplot.subplots()

# visualise demand curve
ax.plot(demand_curve.probability, demand_curve.relative_price)

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.grid()
ax.set_xlabel("Probability of choosing a heat pump")
ax.set_ylabel("Price of a heat pump relative to a gas boiler")

# %%
# Create a population of households with gas boilers and heat pumps.
population = [
    Household(
        heat_demand=12_000,
        discount_rate=0.05,
        demand_curve=demand_curve,
        heat_source=gas_boiler,
    )
    for i in range(9900)
]
# Add some heat pumps
population.extend(
    [
        Household(
            heat_demand=12_000,
            discount_rate=0.05,
            demand_curve=demand_curve,
            heat_source=heat_pump,
        )
        for i in range(100)
    ]
)


# %%
# A model object
class Model:
    def __init__(
        self,
        population: Sequence[Household],
        time_steps: int,
        replacement_fraction: float,
        gas_boiler_upfront: float = 2_500,
        heat_pump_upfront: float = 12_000,
    ):
        self.population = deepcopy(population)
        self.time_steps = time_steps
        self.replacement_fraction = replacement_fraction
        self.gas_boiler_upfront = gas_boiler_upfront
        self.heat_pump_upfront = heat_pump_upfront
        # Only instantiate a random number generator once.
        self._rng = numpy.random.default_rng()

    @property
    def population_size(self) -> int:
        return len(self.population)

    def _get_population_indices(self) -> Sequence[int]:
        return self._rng.choice(
            a=numpy.arange(self.population_size),
            size=int(self.population_size * self.replacement_fraction),
            replace=False,
        )

    def _get_demand_curve_probability_from_relative_price(
        self, demand_curve: DemandCurve, relative_price: float
    ) -> float:
        # Get right index of relative price from observations.
        idx = len(demand_curve.relative_price) - numpy.searchsorted(
            demand_curve.relative_price[::-1], relative_price, side="left"
        )
        # Get bounding price values
        lower, upper = (
            demand_curve.relative_price[idx],
            demand_curve.relative_price[idx - 1],
        )
        # Calculate proportion between left and right values
        adjust = (relative_price - lower) / (upper - lower)
        # adjustment in probability
        prob_adjust = (
            demand_curve.probability[idx] - demand_curve.probability[idx - 1]
        ) * adjust
        # Interpolated probability
        prob = demand_curve.probability[idx] - prob_adjust
        return prob

    def _update_heat_source(self, household) -> Household:
        # NB only update heat source if gas boiler, keep heat pump if pre-existing heat pump
        if household.heat_source.name == "Gas Boiler":
            # Calculate the total annual net present value for a household and get relative price
            gas_boiler_tanpv = household.total_annualised_net_present_value(
                total_upfront_cost=self.gas_boiler_upfront, heat_source=gas_boiler
            )
            heat_pump_tanpv = household.total_annualised_net_present_value(
                total_upfront_cost=self.heat_pump_upfront, heat_source=heat_pump
            )
            relative_price = heat_pump_tanpv - gas_boiler_tanpv
            print(relative_price)
            # Get probability from demand curve and relative price
            prob = self._get_demand_curve_probability_from_relative_price(
                household.demand_curve, relative_price
            )
            # Randomly assign heat source
            if self._rng.random() < prob:
                household.heat_source = heat_pump
            else:
                household.heat_source = gas_boiler
        return household

    def run(self) -> None:
        """Run the model, updating the population in place."""
        for t in range(self.time_steps):
            # Choose households to update at this time step
            indices = self._get_population_indices()
            # Iterate over selected
            new_households = []
            for i in indices:
                candidate = self.population[i]
                # Update household - this could be a function of the household object in future.
                new_households.append(self._update_heat_source(candidate))
            # Update population
            for index, replacement in zip(indices, new_households):
                self.population[index] = replacement
        # Currently updating population in place, so return self.
        return self


# %%
# %%timeit
model = Model(population, 20, 0.04).run()

# %%
f, ax = pyplot.subplots()

ax.hist(hp_uptake, bins=40, density=True, alpha=0.5, color="dodgerblue")
z = gaussian_kde(hp_uptake)(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="dodgerblue")

ax.axvline(sum(hp_uptake) / len(hp_uptake), ymax=0.98, color="indianred")

ax.set_xlabel("Heat Pump Uptake Proportion")
ax.set_ylabel("Density")
ax.set_title("Model Outcome (1000 runs)")

# %%
hp_uptake = {"0.02": [], "0.04": [], "0.06": [], "0.08": [], "0.1": []}

for discount_rate in ["0.02", "0.04", "0.06", "0.08", "0.1"]:
    # Create a population of households with gas boilers and heat pumps.
    population = [
        Household(
            heat_demand=12_000,
            discount_rate=float(discount_rate),
            demand_curve=demand_curve,
            heat_source=gas_boiler,
        )
        for i in range(9900)
    ]
    # Add some heat pumps
    population.extend(
        [
            Household(
                heat_demand=12_000,
                discount_rate=float(discount_rate),
                demand_curve=demand_curve,
                heat_source=heat_pump,
            )
            for i in range(100)
        ]
    )
    for run in range(1000):
        model = Model(population, 20, 0.04).run()
        count_hp = 0
        for hh in model.population:
            if hh.heat_source.name == "Heat Pump":
                count_hp += 1
        hp_uptake[discount_rate].append(count_hp / model.population_size)

# %%
f, ax = pyplot.subplots(figsize=(6, 5))

# 0.02
ax.hist(hp_uptake["0.02"], bins=40, density=True, alpha=0.3, color="#e41a1c")
z = gaussian_kde(hp_uptake["0.02"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="#e41a1c")
ax.axvline(
    sum(hp_uptake["0.02"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#e41a1c",
    label="2%",
)

# 0.04
ax.hist(hp_uptake["0.04"], bins=40, density=True, alpha=0.3, color="#377eb8")
z = gaussian_kde(hp_uptake["0.04"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="#377eb8")
ax.axvline(
    sum(hp_uptake["0.04"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#377eb8",
    label="4%",
)

# 0.06
ax.hist(hp_uptake["0.06"], bins=40, density=True, alpha=0.3, color="#4daf4a")
z = gaussian_kde(hp_uptake["0.06"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="#4daf4a")
ax.axvline(
    sum(hp_uptake["0.06"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#4daf4a",
    label="6%",
)

# 0.08
ax.hist(hp_uptake["0.08"], bins=40, density=True, alpha=0.3, color="#984ea3")
z = gaussian_kde(hp_uptake["0.08"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="#984ea3")
ax.axvline(
    sum(hp_uptake["0.08"]) / len(hp_uptake["0.08"]),
    ymax=0.98,
    color="#984ea3",
    label="8%",
)

# 0.1
ax.hist(hp_uptake["0.1"], bins=40, density=True, alpha=0.3, color="#ff7f00")
z = gaussian_kde(hp_uptake["0.1"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.22, 0.25, 100), z, color="#ff7f00")
ax.axvline(
    sum(hp_uptake["0.1"]) / len(hp_uptake["0.1"]),
    ymax=0.98,
    color="#ff7f00",
    label="10%",
)

ax.set_xlabel("Heat Pump Uptake Proportion")
ax.set_ylabel("Density")
ax.set_title("Model Outcome (1000 runs; 5 Discount Rates)")

ax.legend(title="Discount Rate", loc=2)

# %% [markdown]
# Lower uptake at higher discount rates is a by product of the fact that gas is more expensive in the model and hence gets relatively cheaper at higher rates of discount, which increases the relative price of a heat pump and lowers the probability of adoption.

# %%
# Explore why the rate falls for higher discounts
npv_running_cost_gb = []
npv_running_cost_hp = []

for discount_rate in numpy.linspace(0.01, 0.15, 15):
    gb_hh = Household(
        heat_demand=12_000,
        discount_rate=discount_rate,
        demand_curve=demand_curve,
        heat_source=gas_boiler,
    )

    hp_hh = Household(
        heat_demand=12_000,
        discount_rate=discount_rate,
        demand_curve=demand_curve,
        heat_source=heat_pump,
    )

    npv_running_cost_gb.append(gb_hh.running_cost_net_present_value(gb_hh.heat_source))
    npv_running_cost_hp.append(hp_hh.running_cost_net_present_value(hp_hh.heat_source))

# %%
f, [ax1, ax2] = pyplot.subplots(1, 2, figsize=(12, 5))

# NB running costs are fixed over time.
# NB upfront costs are not discounted as they are upfront.

ax1.plot(numpy.linspace(0.01, 0.15, 15), npv_running_cost_gb, label="Gas Boiler")
ax1.plot(numpy.linspace(0.01, 0.15, 15), npv_running_cost_hp, label="Heat Pump")
ax1.legend()
ax1.set_ylabel("Value")
ax1.set_title("Running cost net present value")
ax1.set_xlabel("Discount Rate")

ax2.plot(
    numpy.linspace(0.01, 0.15, 15),
    numpy.array(npv_running_cost_hp) - numpy.array(npv_running_cost_gb),
)
ax2.set_ylabel("Difference in running cost net present value")
ax2.set_xlabel("Discount Rate")
ax2.set_title("Heat Pump Relative to Gas Boiler")
