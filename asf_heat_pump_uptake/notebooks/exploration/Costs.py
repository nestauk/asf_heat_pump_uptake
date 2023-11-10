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
from typing import Dict

import pandas
import matplotlib.pyplot as plt
import math
import numpy
import statsmodels.api as sm

# %% [markdown]
# # Costs
#
# The estimation of cost at any given point is a combination of:
# - material/financial costs of: installation of a new heat source; running costs of the heat source; service and maintenance of the heat source.
# - Behavioural modifiers of cost: deadline effect of ban; future vs. present focus (status quo bias); effect of pro-environmental attitude as a driver of early adoption.
#
# Treat upfront costs differently from running costs - this will involve a discount cost and will be the behavioural basis from which people can customise.
#
# Low and high discount rate groups. (5-10%) representing maybe present and future looking people.
#
# Offer of finance - transforming upfront cost into a running cost. E.g. loans for upfront cost.
#
# Demand curve - price ~ quantity (how many people take things up). Turn prices into uptake. Segment population into a few groups with different demand curves.
#
# # Key Behaviours Affecting Costs
#
#

# %%
MEDIAN_COST_GBP_HEAT_PUMP_AIR_SOURCE: Dict[int, int] = {
    # Source: RHI December 2020 Data
    # Adjusted for monotonicity: cost at each capacity >= highest trailing value
    # These values incorporate installation costs
    1: 1500,
    2: 3000,
    3: 4500,
    4: 6000,
    5: 7500,
    6: 7500,
    7: 8050,
    8: 9200,
    9: 10350,
    10: 11500,
    11: 11500,
    12: 11500,
    13: 12350,
    14: 13300,
    15: 14250,
    16: 14250,
    17: 14250,
    18: 14580,
    19: 15390,
    20: 16200,
}

# %%
MEDIAN_COST_GBP_HEAT_PUMP_GROUND_SOURCE: Dict[int, int] = {
    # Adjusted for monotonicity: cost at each capacity >= highest trailing value
    # These values incorporate installation costs
    1: 1800,
    2: 3600,
    3: 5400,
    4: 7200,
    5: 9000,
    6: 10920,
    7: 12740,
    8: 14560,
    9: 16380,
    10: 18200,
    11: 18200,
    12: 18840,
    13: 20410,
    14: 21980,
    15: 23550,
    16: 23550,
    17: 24990,
    18: 26460,
    19: 27930,
    20: 29400,
    21: 29400,
    22: 29400,
    23: 30590,
    24: 31920,
    25: 33250,
}

# %%
BOILER_INSTALLATION_COST_GBP = 1_000

MEAN_COST_GBP_BOILER_GAS: Dict[str, int] = {
    # Source: https://www.boilerguide.co.uk/articles/what-size-boiler-needed
    "PropertySize.SMALL": 2277 + BOILER_INSTALLATION_COST_GBP,
    "PropertySize.MEDIUM": 2347 + BOILER_INSTALLATION_COST_GBP,
    "PropertySize.LARGE": 2476 + BOILER_INSTALLATION_COST_GBP,
}

# %%
# SOURCE: https://webarchive.nationalarchives.gov.uk/ukgwa/20121205193015/http:/www.decc.gov.uk/assets/decc/what%20we%20do/uk%20energy%20supply/energy%20mix/distributed%20energy%20heat/1467-potential-costs-district-heating-network.pdf
# "First-time" installation costs (e.g. pipework, radiator upgrades, boreholes) are approximately 10% of total costs for ASHP, and 50% of total costs of a GSHP
HEAT_PUMP_AIR_SOURCE_REINSTALL_DISCOUNT = 0.1
HEAT_PUMP_GROUND_SOURCE_REINSTALL_DISCOUNT = 0.5

# Source: Distribution based on values in https://www.theccc.org.uk/publication/analysis-of-alternative-uk-heat-decarbonisation-pathways/
DECOMMISSIONING_COST_MIN, DECOMMISSIONING_COST_MAX = 500, 2_000

# %% [markdown]
# ## Calculating Heating System Costs
# ### CNZ ABM
#
# For a choice set of heating systems:
#
# `get_total_heating_system_costs()`
# 1. calculate 'unit and install costs'
# 2. calculate 'fuel costs'
# 3. calculate subsidies
#
# `get_unit_and_install_costs(heating_system)`
# 1. calculate decommisioning costs (if changing heating system). Random value between £500 and £2000.
# 2. calculate the unit and install cost according to heating type
#     a. air/ground source heat pump median cost based on estimated capacity from floor area.
#         &nbsp; &nbsp; &nbsp; &nbsp;- If air source, apply discount factor of 30% in 2023 and 60% in 2025. No discount for GSHP.
#         &nbsp; &nbsp; &nbsp; &nbsp;- Apply reinstall discount if existing heat source is a heat pump.
#     b. gas boiler - mean cost based on property size (small; medium; large)
#     c. oil boiler - mean cost based on property size (small; medium; large)
#     d. electric boiler - mean cost based on property size (small; medium; large)
#
# `get_heating_fuel_costs(heating_system)`
# (Only calculated for owner occupiers as tenants costs will not usually be a concern for landlords.)
# 1. calculate heating fuel costs 'net present value'
#     a. Get coefficient of performance scale factor (efficiency of current system / efficiency of system of interest).
#     b. Get scaled annual heating demand (system specific demand relative to current system)
#     c. Calculate annual heating bill - demand x fuel price
#     d. return discount annual cash flow (household discount rate, annual heating bill, household number of years look ahead)
#     &nbsp; &nbsp; &nbsp; &nbsp;- $\sum_{t} \frac{\textrm{annual heating bill}}{(1 + \textrm{discount_rate})^{t}}$
#
#
# NB discount rate is based on an agents wealth, with a maximum value of 1. It's value is calculated from a weibull distribution with alpha = 0.8 and beta = 0.165 and percentile = 1 - wealth percentile. Given as: $ \beta \times (- \log (1- percentile))^{1/alpha}$
#
# Wealth percentile is set from the house price, against from a weibull distribution (min = 0.001, max = 0.999), with an alpha = 1.61 and beta = 280,000.
#
# Subsidies are estimated if 'boiler upgrade scheme' (BUS) and/or 'renewable heat incentive' (RHI) are in the model.
#
# unit and install costs, fuel costs net present value, and (-1 \*) subsidies are returned.
#
# These costs are then summed to a single value for each heating system.
#
# `choose_heating_system(costs, hasslefactor)`
# 1. for each heating system, work out the weight as a function of the renovation budget.
#     a. Get each cost as a proportion of the budget.
#     b. Calculate the weight as 1 / exp(a)
#     c. If the heating system is a hassle, reduce the weight according to the hassle factor ()
# 2. If all the weights are highly unaffordable (> 10x the budget) 'repair' the current heating system by returning that.
# 3. Otherwise, randomly choose a new heating system using the calculated weights.
#
#
# The renovation budget for heating is 20% of the total rennovation budget, which is a value drawn from a weibull distribution with alpha=0.55, beta=21994 and based on the agents wealth percentile (see above).
#
# *Discount rates*: A discount rate is an interest rate used to determine the present value of a future cash flow. In other words, the higher the discount rate, the more a household values money today compared to money in the future.

# %%
GB_PROPERTY_VALUE_WEIBULL_ALPHA = 1.61
GB_PROPERTY_VALUE_WEIBULL_BETA = 280_000
DISCOUNT_RATE_WEIBULL_ALPHA = 0.8
DISCOUNT_RATE_WEIBULL_BETA = 0.165


def get_weibull_percentile_from_value(
    alpha: float, beta: float, input_value: float
) -> float:
    return 1 - math.exp(-((input_value / beta) ** alpha))


def get_weibull_value_from_percentile(
    alpha: float, beta: float, percentile: float
) -> float:
    return beta * (-math.log(1 - percentile)) ** (1 / alpha)


def wealth_percentile(property_value_gbp) -> float:
    PERCENTILE_FLOOR = 0.001
    PERCENTILE_CAP = 0.999
    percentile = get_weibull_percentile_from_value(
        GB_PROPERTY_VALUE_WEIBULL_ALPHA,
        GB_PROPERTY_VALUE_WEIBULL_BETA,
        property_value_gbp,
    )
    return min(max(percentile, PERCENTILE_FLOOR), PERCENTILE_CAP)


def discount_rate(wealth_percentile) -> float:
    DISCOUNT_RATE_CAP = 1.0
    percentile = get_weibull_value_from_percentile(
        DISCOUNT_RATE_WEIBULL_ALPHA,
        DISCOUNT_RATE_WEIBULL_BETA,
        1 - wealth_percentile,
    )
    return min(percentile, DISCOUNT_RATE_CAP)


# %%
f, ax = plt.subplots(figsize=(8, 5))

# Property value -> Wealth Percentile
property_value = numpy.linspace(0, 1_000_000, 1_000)

ax.plot(property_value, numpy.vectorize(wealth_percentile)(property_value) * 100)
ax.set_xlabel("Property Value (GBP)")
ax.set_ylabel("Wealth Percentile")
ax.grid()
ax.set_title("CNZ Agent Wealth Percentiles for given Property Values")

# %%
# Discount Rates
f, ax = plt.subplots(figsize=(8, 5))

# Wealth Percentile -> discount rate
wealth_percentile = numpy.linspace(0.001, 0.999, 999)

ax.plot(wealth_percentile * 100, numpy.vectorize(discount_rate)(wealth_percentile))
ax.set_ylabel("Discount Rate")
ax.set_xlabel("Wealth Percentile")
ax.grid()
ax.set_title("CNZ Agent Discount Rates for given Wealth Percentiles")

# %%
# calculate heating fuel costs 'net present value'

# Scenario 1a - gas boiler for gas boiler - median person (wealth 0.5), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 0.92
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.0589  # gas price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.5)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Scenario 2a - gas boiler for gas boiler - Wealthy person (wealth 0.9), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 0.92
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.0589  # gas price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.9)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Scenario 3a - gas boiler for gas boiler - Poor person (wealth 0.2), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 0.92
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.0589  # gas price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.2)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Scenario 1b - gas boiler for ASHP - median person (wealth 0.5), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 3
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.1494  # electricity price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.5)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Scenario 2b - gas boiler for gas boiler - Wealthy person (wealth 0.9), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 3
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.1494  # electricity price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.9)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Scenario 3b - gas boiler for gas boiler - Poor person (wealth 0.2), 3 year lookahead. 100m2 medium house.
COP_SCALE_FACTOR = 0.92 / 3
annual_heating_demand = 12_000 * COP_SCALE_FACTOR
annual_heating_bill = annual_heating_demand * 0.1494  # electricity price
discount_annual_cash_flow = sum(
    [annual_heating_bill / (1 + discount_rate(0.2)) ** t for t in range(3)]
)
print(f"£{round(discount_annual_cash_flow, 2)}")

# %%
# Gas to electricty price ratio
0.1494 / 0.0589

# %%
FUEL_KWH_TO_HEAT_KWH: Dict[str, float] = {
    # The conversion factor between 1kWh of fuel and useful heat. For example:
    # Gas Boilers ~ 0.9, since 1kWh of gas produces ~0.9kWh of heat (due to inefficiencies in the boiler)
    "BOILER_GAS": 0.92,
    "BOILER_OIL": 0.92,
    "BOILER_ELECTRIC": 0.995,
    "HEAT_PUMP_AIR_SOURCE": 3,
    "HEAT_PUMP_GROUND_SOURCE": 4,
}

# %%
# Scale factor is inferred from general relationship between estimated floor area and kW capacity
# https://www.boilerguide.co.uk/articles/size-heat-pump-need (see table)
# https://www.imsheatpumps.co.uk/blog/what-size-heat-pump-do-i-need-for-my-house/
# https://www.homeheatingguide.co.uk/renewables-advice/air-source-heat-pumps-a-sizing-guide
HEAT_PUMP_CAPACITY_SCALE_FACTOR = {
    "HEAT_PUMP_AIR_SOURCE": 0.1,
    "HEAT_PUMP_GROUND_SOURCE": 0.08,
}

MAX_HEAT_PUMP_CAPACITY_KW = {
    "HEAT_PUMP_AIR_SOURCE": 20.0,
    "HEAT_PUMP_GROUND_SOURCE": 25.0,
}

MIN_HEAT_PUMP_CAPACITY_KW = {
    "HEAT_PUMP_AIR_SOURCE": 4.0,
    "HEAT_PUMP_GROUND_SOURCE": 4.0,
}


def compute_heat_pump_capacity_kw(total_floor_area_m2, heat_pump_type) -> int:
    """Heat Pump Size in Kw is a function of total floor area. Min kW=4, max kW=20or25"""
    capacity_kw = HEAT_PUMP_CAPACITY_SCALE_FACTOR[heat_pump_type] * total_floor_area_m2
    return math.ceil(
        min(
            max(capacity_kw, MIN_HEAT_PUMP_CAPACITY_KW[heat_pump_type]),
            MAX_HEAT_PUMP_CAPACITY_KW[heat_pump_type],
        )
    )


# %% [markdown]
# ## MCS Data

# %%
mcs_epc = pandas.read_parquet(
    "s3://asf-daps/lakehouse/processed/mcs/mcs_installations_epc_dedupl_most_relevant_20230619-0.parquet"
)

# %% [markdown]
# ## Air Source

# %%
air_source = mcs_epc.loc[
    lambda df: (df["tech_type"] == "Air Source Heat Pump")
    & (df["capacity"].notna())
    & df["capacity"].between(0.1, 26)
]

# %%
model_ols = sm.OLS.from_formula("cost ~ capacity", data=air_source).fit()
model_ols_log = sm.OLS.from_formula("cost ~ numpy.log(capacity)", data=air_source).fit()

# %%
model_ols_boxcox = sm.OLS.from_formula(
    "cost ~ transform(capacity)", data=air_source
).fit()

# %%
model_quant = sm.QuantReg.from_formula("cost ~ capacity", data=air_source).fit()
model_quant_log = sm.QuantReg.from_formula(
    "cost ~ numpy.log(capacity)", data=air_source
).fit()

# %%
f, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    air_source.loc[lambda df: df["cost"] < 50000, "capacity"],
    air_source.loc[lambda df: df["cost"] < 50000, "cost"],
    marker=".",
)

ax.plot(
    list(range(1, 26)),
    model_ols.predict(pandas.Series(range(1, 26), name="capacity")),
    color="r",
)
ax.plot(
    list(range(1, 26)),
    model_quant.predict(pandas.Series(range(1, 26), name="capacity")),
    color="g",
)

ax.plot(
    list(range(1, 26)),
    model_ols_log.predict(pandas.Series(range(1, 26), name="capacity")),
    color="y",
)
ax.plot(
    list(range(1, 26)),
    model_quant_log.predict(pandas.Series(range(1, 26), name="capacity")),
    color="b",
)

# %%
f, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    list(range(1, 21)),
    list(map(MEDIAN_COST_GBP_HEAT_PUMP_AIR_SOURCE.get, range(1, 21))),
)

ax.plot(
    list(range(1, 21)),
    model_quant.predict(pandas.Series(range(1, 21), name="capacity")),
    color="g",
)
ax.plot(
    list(range(1, 21)),
    model_ols.predict(pandas.Series(range(1, 21), name="capacity")),
    color="r",
)

ax.plot(
    list(range(1, 21)),
    model_quant_log.predict(pandas.Series(range(1, 21), name="capacity")),
    color="b",
)
ax.plot(
    list(range(1, 21)),
    model_ols_log.predict(pandas.Series(range(1, 21), name="capacity")),
    color="y",
)

ax.set_xticks(range(1, 21))
ax.grid()
ax.set_xlabel("Installed Capacity (kW)")
ax.set_ylabel("Median Cost (£)")
ax.set_title("Air Source Heat Pumps")

# %%
model_ols = sm.OLS.from_formula(
    "cost ~ capacity + C(commission_year)", data=air_source
).fit()

# %%
cost_year = model_ols.predict(
    pandas.DataFrame(
        {
            "capacity": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "commission_year": [
                2009,
                2010,
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
                2022,
            ],
        }
    )
)

# %%
f, ax = plt.subplots()

ax.plot(
    [
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
    ],
    cost_year,
)

ax.set_ylabel("Mean Cost (£)")
ax.set_xlabel("Year")
ax.set_title("Mean Cost of 6kW Air Source (unadjusted)")
ax.grid()

# %%
model_ols.summary()

# %%
air_source.columns

# %%
air_source.commission_year.min()

# %% [markdown]
# ## Ground Source

# %%
ground_source = mcs_epc.loc[
    lambda df: (df["tech_type"] == "Ground/Water Source Heat Pump")
    & (df["capacity"].notna())
    & df["capacity"].between(0.1, 50)
]

# %%
model_ols = sm.OLS.from_formula("cost ~ capacity", data=ground_source).fit()
model_quant = sm.QuantReg.from_formula("cost ~ capacity", data=ground_source).fit()

# %%
f, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    ground_source.loc[lambda df: df["cost"] < 100000, "capacity"],
    ground_source.loc[lambda df: df["cost"] < 100000, "cost"],
    marker=".",
)

ax.plot(
    list(range(1, 50)),
    model_ols.predict(pandas.Series(range(1, 50), name="capacity")),
    color="r",
)
ax.plot(
    list(range(1, 50)),
    model_quant.predict(pandas.Series(range(1, 50), name="capacity")),
    color="g",
)

# %%
f, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    list(range(1, 26)),
    list(map(MEDIAN_COST_GBP_HEAT_PUMP_GROUND_SOURCE.get, range(1, 26))),
)

ax.plot(
    list(range(1, 26)),
    model_ols.predict(pandas.Series(range(1, 26), name="capacity")),
    color="r",
)
ax.plot(
    list(range(1, 26)),
    model_quant.predict(pandas.Series(range(1, 26), name="capacity")),
    color="g",
)

ax.set_xticks(range(1, 26))
ax.grid()
ax.set_xlabel("Installed Capacity (kW)")
ax.set_ylabel("Median Cost (£)")
ax.set_title("Ground Source Heat Pumps")
