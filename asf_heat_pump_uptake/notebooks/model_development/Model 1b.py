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
from dataclasses import dataclass, field
from typing import Sequence
from copy import deepcopy
from matplotlib import pyplot
from scipy.stats import gaussian_kde, weibull_min
import pandas


# %% [markdown]
# ## Model 1b
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
    failure_distribution: weibull_min
    install_year: int  # New in Model 1b


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
    heat_source_history: list = field(default_factory=list)

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

    def is_heat_source_operational(self, current_year, rng) -> bool:  # New in Model 1b
        """Check if heat source is operational based on boiler age"""
        heat_source_age = current_year - self.heat_source.install_year
        # This is the hazard rate
        failure_prob = self.heat_source.failure_distribution.pdf(
            heat_source_age
        ) / self.heat_source.failure_distribution.sf(heat_source_age)
        if rng.random() < failure_prob:
            heat_source_status = False
        else:
            heat_source_status = True
        return heat_source_status

    def update_heat_source(self, new_heat_source) -> None:  # New in Model 1b
        """Set current heat source to new_heat_source and archive old heat source."""
        self.heat_source_history.append(self.heat_source)
        self.heat_source = new_heat_source
        return None


# %%
# Fuels
natural_gas = Fuel(name="Natural Gas", price=5)
electricity = Fuel(name="Electricity", price=15)


# This effectively defines a gas boiler and heat pump archetype.
@dataclass
class GasBoiler(HeatSource):
    name: str = "Gas Boiler"
    lifespan: int = 15
    efficiency: float = 0.85
    fuel: Fuel = natural_gas
    failure_distribution: weibull_min = weibull_min(c=2.2, loc=2, scale=13)
    install_year: int = None  # Requires setting


@dataclass
class HeatPump(HeatSource):
    name: str = "Heat Pump"
    lifespan: int = 20
    efficiency: float = 3.0
    fuel: Fuel = electricity
    failure_distribution: weibull_min = weibull_min(c=2.5, loc=4, scale=18)
    install_year: int = None  # Requires setting


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
hazard_rate_hp = lambda year: HeatPump().failure_distribution.pdf(
    year
) / HeatPump().failure_distribution.sf(year)
hazard_rate_gb = lambda year: GasBoiler().failure_distribution.pdf(
    year
) / GasBoiler().failure_distribution.sf(year)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

ax.plot(
    numpy.linspace(0, 50, 51),
    [hazard_rate_hp(x) for x in numpy.linspace(0, 50, 51)],
    label="Heat Pump",
)
ax.plot(
    numpy.linspace(0, 50, 51),
    [hazard_rate_gb(x) for x in numpy.linspace(0, 50, 51)],
    label="Gas Boiler",
)

ax.set_xlabel("Age (Years)")
ax.set_ylabel("Hazard rate")
ax.grid()
ax.legend()

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

ax.plot(
    numpy.linspace(0, 40, 1000),
    GasBoiler().failure_distribution.sf(numpy.linspace(0, 40, 1000)),
    label="Gas Boiler",
)
ax.plot(
    numpy.linspace(0, 40, 1000),
    HeatPump().failure_distribution.sf(numpy.linspace(0, 40, 1000)),
    label="Heat Pump",
)

ax.grid()
ax.set_xlabel("Age of Heat Source")
ax.set_ylabel("Survival Probability")
ax.legend()

# %%
f, ax = pyplot.subplots()

# visualise demand curve
ax.plot(demand_curve.probability, demand_curve.relative_price)

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.grid()
ax.set_xlabel("Probability of choosing a heat pump")
ax.set_ylabel("Price of a heat pump relative to a gas boiler")

# %%
# Distribution based on new installs
gas_boiler_year = (
    lambda x: 1036612.16083916
    + 276000.98901099 * (-335.83333333 + 0.16666667 * x)
    - 35931.77622378 * (-335.83333333 + 0.16666667 * x) ** 2
)

# %%
counts_by_year = gas_boiler_year(numpy.arange(2003, 2023))
prop_by_year = counts_by_year / sum(counts_by_year)

# %%
f, ax = pyplot.subplots(figsize=(8, 4))

ax.bar(numpy.arange(2003, 2023), prop_by_year)

ax.set_xticks(numpy.arange(2003, 2023, 2))
ax.set_xlabel("Install Year")
ax.set_ylabel("Proportion of population")

ax.grid(axis="y")
ax.set_axisbelow(True)

# %%
# Create a population (n=10_000) of households with gas boilers and heat pumps.
population = []
for year, count in zip(
    numpy.arange(2003, 2023), (prop_by_year * 9898).round(0).astype(int)
):
    population.extend(
        [
            Household(
                heat_demand=12_000,
                discount_rate=0.05,
                demand_curve=demand_curve,
                heat_source=GasBoiler(install_year=year),
            )
            for _ in range(count)
        ]
    )
for year, count in zip(numpy.arange(2018, 2023), [20, 20, 20, 20, 20]):
    population.extend(
        [
            Household(
                heat_demand=12_000,
                discount_rate=0.05,
                demand_curve=demand_curve,
                heat_source=HeatPump(install_year=year),
            )
            for _ in range(count)
        ]
    )


# %%
# A model object
class Model:
    def __init__(
        self,
        population: Sequence[Household],
        start_year: int,
        end_year: int,
        time_step: int,
        gas_boiler_upfront: float = 2_500,
        heat_pump_upfront: float = 12_000,
    ):
        self.population = deepcopy(population)
        self.start_year = start_year
        self.end_year = end_year
        self.time_step = time_step
        self.gas_boiler_upfront = gas_boiler_upfront
        self.heat_pump_upfront = heat_pump_upfront
        self.replacement_count = []
        # Only instantiate a random number generator once.
        self._rng = numpy.random.default_rng()

    @property
    def population_size(self) -> int:
        return len(self.population)

    def _get_population_indices(
        self, current_year
    ) -> Sequence[int]:  # this changes to heat source failures.
        # check all heat sources to see if they are operational, record those that are not.
        indices = []
        for idx, hh in enumerate(self.population):
            if not hh.is_heat_source_operational(current_year, self._rng):
                indices.append(idx)
        return indices

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

    def _make_replacement_decision(self, household, year) -> Household:
        # NB only update heat source if gas boiler, keep heat pump if pre-existing heat pump
        if household.heat_source.name == "Gas Boiler":
            # Calculate the total annual net present value for a household and get relative price
            gas_boiler_tanpv = household.total_annualised_net_present_value(
                total_upfront_cost=self.gas_boiler_upfront, heat_source=GasBoiler()
            )
            heat_pump_tanpv = household.total_annualised_net_present_value(
                total_upfront_cost=self.heat_pump_upfront, heat_source=HeatPump()
            )
            relative_price = heat_pump_tanpv - gas_boiler_tanpv
            # Get probability from demand curve and relative price
            prob = self._get_demand_curve_probability_from_relative_price(
                household.demand_curve, relative_price
            )
            # Randomly assign heat source
            if self._rng.random() <= prob:
                household.update_heat_source(HeatPump(install_year=year))
            else:
                household.update_heat_source(GasBoiler(install_year=year))
        elif household.heat_source.name == "Heat Pump":
            # Just get new heat pump
            household.update_heat_source(HeatPump(install_year=year))
        return household

    def run(self) -> None:
        """Run the model, updating the population in place."""
        for t in range(self.start_year, self.end_year, self.time_step):
            # Choose households to update at this year
            indices = self._get_population_indices(t)
            # print(len(indices))
            self.replacement_count.append(len(indices))
            # Iterate over selected
            for i in indices:
                # Update household
                self.population[i] = self._make_replacement_decision(
                    self.population[i], t
                )
        # Currently updating population in place, so return self.
        return self


# %%
model = Model(
    population,
    start_year=2023,
    end_year=2051,
    time_step=1,
    gas_boiler_upfront=2_500,
    heat_pump_upfront=12_000,
).run()

# %%
gas_lifespan = []
heat_pump_lifespan = []
for hh in model.population:
    if len(hh.heat_source_history) > 0:
        if len(hh.heat_source_history) > 1:
            if hh.heat_source_history[0].name == "Gas Boiler":
                gas_lifespan.append(
                    hh.heat_source_history[1].install_year
                    - hh.heat_source_history[0].install_year
                )
            if hh.heat_source_history[0].name == "Heat Pump":
                heat_pump_lifespan.append(
                    hh.heat_source_history[1].install_year
                    - hh.heat_source_history[0].install_year
                )
        elif len(hh.heat_source_history) == 1:
            if hh.heat_source_history[0].name == "Gas Boiler":
                gas_lifespan.append(
                    hh.heat_source.install_year - hh.heat_source_history[0].install_year
                )
            if hh.heat_source_history[0].name == "Heat Pump":
                heat_pump_lifespan.append(
                    hh.heat_source.install_year - hh.heat_source_history[0].install_year
                )

# %%
numpy.mean(gas_lifespan)

# %%
numpy.mean(heat_pump_lifespan)

# %% [markdown]
# ### Replacement Rate

# %%
replacement_count = []
for run in range(100):
    model = Model(population, start_year=2023, end_year=2051, time_step=1).run()
    replacement_count.append(model.replacement_count)

# %%
replacement_count = numpy.column_stack(replacement_count)

# %%
mean = replacement_count.mean(axis=1) / 10000
sd = replacement_count.std(axis=1) / 10000
upper = mean + 1.96 * sd
lower = mean - 1.96 * sd

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

ax.plot(mean)
ax.fill_between(range(len(mean)), lower, upper, alpha=0.5)
ax.set_ylabel("Annual replacement rate")
ax.set_xlabel("Model year")


# %% [markdown]
# ### Annual Uptake


# %%
def get_heat_source_at_given_year(household: Household, year: float):
    """Derive the type of home heating for a given household in a given year."""
    heat_sources = household.heat_source_history + [household.heat_source]
    name = None
    index = -1
    try:
        while not name:
            if year >= heat_sources[index].install_year:
                name = heat_sources[index].name
            else:
                index -= 1
    except IndexError:
        pass
    return name


def create_uptake_dataframe(model):
    return pandas.concat(
        [
            pandas.Series(
                [
                    get_heat_source_at_given_year(household=hh, year=year)
                    for hh in model.population
                ],
                name=f"year_{year}",
            )
            for year in range(2023, 2051)
        ],
        axis=1,
    )


# %%
uptake = create_uptake_dataframe(model)

# %%
tidy_uptake = (
    pandas.concat(
        [
            uptake[f"year_{year}"].value_counts().rename(f"{year}")
            for year in range(2023, 2051)
        ],
        axis=1,
    )
    .reset_index()
    .melt(id_vars="index", var_name="year")
)

# %%
tidy_uptake["prop"] = tidy_uptake["value"] / 9_999

# %%
f, ax = pyplot.subplots(figsize=(12, 4))

ax.plot(
    tidy_uptake.loc[lambda df: df["index"] == "Gas Boiler", "year"],
    tidy_uptake.loc[lambda df: df["index"] == "Gas Boiler", "prop"],
)

ax.plot(
    tidy_uptake.loc[lambda df: df["index"] == "Heat Pump", "year"],
    tidy_uptake.loc[lambda df: df["index"] == "Heat Pump", "prop"],
)

ax.grid()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# %%
uptake_runs = []
for run in range(100):
    model = Model(population, start_year=2023, end_year=2051, time_step=1).run()
    uptake = create_uptake_dataframe(model)
    uptake_runs.append(
        pandas.concat(
            [
                uptake[f"year_{year}"].value_counts().rename(f"{year}")
                for year in range(2023, 2051)
            ],
            axis=1,
        )
        .reset_index()
        .melt(id_vars="index", var_name="year")
    )
    print(run)

# %%
import pickle

# pickle.dump(uptake_runs, open("./uptake_Runs.pkl", 'wb'))
pickleFile = open("./uptake_Runs.pkl", "rb")
uptake_runs = pickle.load(pickleFile)

# %%
# Get mean of runs
mean_gb = pandas.concat(
    [df.loc[lambda df: df["index"] == "Gas Boiler"] for df in uptake_runs], axis=1
)["value"].mean(axis=1)
std_gb = pandas.concat(
    [df.loc[lambda df: df["index"] == "Gas Boiler"] for df in uptake_runs], axis=1
)["value"].std(axis=1)
upper_gb = (mean_gb + 1.96 * std_gb) / 9999
lower_gb = (mean_gb - 1.96 * std_gb) / 9999
mean_gb /= 9999

mean_hp = pandas.concat(
    [df.loc[lambda df: df["index"] == "Heat Pump"] for df in uptake_runs], axis=1
)["value"].mean(axis=1)
std_hp = pandas.concat(
    [df.loc[lambda df: df["index"] == "Heat Pump"] for df in uptake_runs], axis=1
)["value"].std(axis=1)
upper_hp = (mean_hp + 1.96 * std_hp) / 9999
lower_hp = (mean_hp - 1.96 * std_hp) / 9999
mean_hp /= 9999

# %%
f, ax = pyplot.subplots(figsize=(8, 4))

ax.plot(range(len(mean_gb)), mean_gb, label="Gas Boiler")
ax.fill_between(range(len(mean_gb)), lower_gb, upper_gb, alpha=0.5)
ax.plot(range(len(mean_hp)), mean_hp, label="Heat Pump")
ax.fill_between(range(len(mean_hp)), lower_hp, upper_hp, alpha=0.5)

ax.set_ylabel("Uptake Proportion")
ax.set_xlabel("Model year")
ax.grid()
ax.legend()

# %% [markdown]
# ## Test Discount Rates

# %%
hp_uptake = {"0.02": [], "0.04": [], "0.06": [], "0.08": [], "0.1": []}

for discount_rate in ["0.02", "0.04", "0.06", "0.08", "0.1"]:
    # Create a population of households with gas boilers and heat pumps.
    population = []
    for year, count in zip(
        numpy.arange(2003, 2023), (prop_by_year * 9899).round(0).astype(int)
    ):
        population.extend(
            [
                Household(
                    heat_demand=12_000,
                    discount_rate=float(discount_rate),
                    demand_curve=demand_curve,
                    heat_source=GasBoiler(install_year=year),
                )
                for _ in range(count)
            ]
        )
        for year, count in zip(numpy.arange(2018, 2023), [20, 20, 20, 20, 20]):
            population.extend(
                [
                    Household(
                        heat_demand=12_000,
                        discount_rate=float(discount_rate),
                        demand_curve=demand_curve,
                        heat_source=HeatPump(install_year=year),
                    )
                    for _ in range(count)
                ]
            )

    for run in range(100):
        model = Model(
            population,
            start_year=2023,
            end_year=2051,
            time_step=1,
            gas_boiler_upfront=2_500,
            heat_pump_upfront=12_000,
        ).run()
        count_hp = 0
        for hh in model.population:
            if hh.heat_source.name == "Heat Pump":
                count_hp += 1
        hp_uptake[discount_rate].append(count_hp / model.population_size)
        print(f"{discount_rate} -- {run}")

# %%
import pickle

# pickle.dump(hp_uptake, open("./hp_uptake.pkl", 'wb'))
pickleFile = open("./hp_uptake.pkl", "rb")
hp_uptake = pickle.load(pickleFile)

# %%
max(hp_uptake["0.06"])

# %%
f, ax = pyplot.subplots(figsize=(6, 5))

# 0.02
ax.hist(hp_uptake["0.02"], bins=10, density=True, alpha=0.3, color="#e41a1c")
z = gaussian_kde(hp_uptake["0.02"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.60, 0.65, 100), z, color="#e41a1c")
ax.axvline(
    sum(hp_uptake["0.02"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#e41a1c",
    label="2%",
)

# 0.04
ax.hist(hp_uptake["0.04"], bins=10, density=True, alpha=0.3, color="#377eb8")
z = gaussian_kde(hp_uptake["0.04"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.60, 0.65, 100), z, color="#377eb8")
ax.axvline(
    sum(hp_uptake["0.04"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#377eb8",
    label="4%",
)

# 0.06
ax.hist(hp_uptake["0.06"], bins=10, density=True, alpha=0.3, color="#4daf4a")
z = gaussian_kde(hp_uptake["0.06"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.60, 0.65, 100), z, color="#4daf4a")
ax.axvline(
    sum(hp_uptake["0.06"]) / len(hp_uptake["0.02"]),
    ymax=0.98,
    color="#4daf4a",
    label="6%",
)

# 0.08
ax.hist(hp_uptake["0.08"], bins=10, density=True, alpha=0.3, color="#984ea3")
z = gaussian_kde(hp_uptake["0.08"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.60, 0.65, 100), z, color="#984ea3")
ax.axvline(
    sum(hp_uptake["0.08"]) / len(hp_uptake["0.08"]),
    ymax=0.98,
    color="#984ea3",
    label="8%",
)

# 0.1
ax.hist(hp_uptake["0.1"], bins=10, density=True, alpha=0.3, color="#ff7f00")
z = gaussian_kde(hp_uptake["0.1"])(numpy.linspace(0.22, 0.25, 100))
ax.plot(numpy.linspace(0.60, 0.65, 100), z, color="#ff7f00")
ax.axvline(
    sum(hp_uptake["0.1"]) / len(hp_uptake["0.1"]),
    ymax=0.98,
    color="#ff7f00",
    label="10%",
)

ax.set_xlabel("Heat Pump Uptake Proportion")
ax.set_ylabel("Density")
ax.set_title("Model Outcome (100 runs; 5 Discount Rates)")

ax.legend(title="Discount Rate", loc=2)
