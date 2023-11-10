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
import pandas
from matplotlib import pyplot
import numpy
from scipy.stats import weibull_min

# %%
## Gas Safe Register Data
notifications = pandas.DataFrame(
    data=[
        [2009, 744_775],
        [2010, 825_015],
        [2011, 774_424],
        [2012, 782_688],
        [2013, 1_029_997],
        [2014, 997_553],
        [2015, 1_048_419],
        [2016, 1_108_621],
        [2017, 1_093_030],
        [2018, 1_160_251],
        [2019, 1_232_618],
        [2020, 1_218_316],
        [2021, 1_278_596],
    ],
    columns=["year", "count"],
)

# %%
# House price crash? decline in construction activity? financial crisis? followed by catch-up.
# Growth in households.

f, ax = pyplot.subplots(figsize=(8, 4))

ax.plot(notifications["year"], notifications["count"])
ax.set_xlabel("Year")
ax.set_ylabel("Count of Notifications")
ax.set_title(
    "Notifications received for installation of domestic gas fired boilers\nEngland and Wales",
    loc="left",
)

# %%
poly = numpy.polynomial.Polynomial([2, 1]).fit(
    x=notifications["year"], y=notifications["count"], deg=2
)

# %%
# House price crash? decline in construction activity? financial crisis? followed by catch-up.
# Growth in households.

f, ax = pyplot.subplots(figsize=(8, 4))

ax.plot(notifications["year"], notifications["count"])
ax.set_xlabel("Year")
ax.set_ylabel("Count of Notifications")
ax.set_title(
    "Notifications received for installation of domestic gas fired boilers\nEngland and Wales",
    loc="left",
)

ax.plot(numpy.linspace(2003, 2023, 1000), poly(numpy.linspace(2003, 2023, 1000)))

# %%
gas_boiler_year = (
    lambda x: 1036612.16083916
    + 276000.98901099 * (-335.83333333 + 0.16666667 * x)
    - 35931.77622378 * (-335.83333333 + 0.16666667 * x) ** 2
)

# %%
counts_by_year = gas_boiler_year(numpy.arange(2008, 2023))
prop_by_year = counts_by_year / sum(counts_by_year)

# %%
boiler_ages = pandas.Series(
    numpy.repeat(numpy.arange(2008, 2023), (prop_by_year * 9898).round(0).astype(int)),
    name="start_year",
).to_frame()

# %%
heat_pump_ages = pandas.Series(
    numpy.repeat([2020, 2021, 2022], [2000, 2000, 2000]), name="start_year"
).to_frame()

# %%
dist_gb = weibull_min(c=2.2, loc=2, scale=13)
hazard_rate_gb = lambda year: dist_gb.pdf(year) / dist_gb.sf(year)
dist_hp = weibull_min(c=2.5, loc=4, scale=18)
hazard_rate_hp = lambda year: dist_hp.pdf(year) / dist_hp.sf(year)

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
rng = numpy.random.default_rng()


def estimate_failure_year(x, dist):
    year = 2023
    operational = True
    while operational:
        hazard_rate = dist.pdf(year - x) / dist.sf(year - x)
        if rng.random() < hazard_rate:
            operational = False
        year += 1
    return year


# %%
boiler_ages["first_failure"] = boiler_ages.apply(
    lambda x: estimate_failure_year(x["start_year"], dist_gb), axis=1
)
boiler_ages["second_failure"] = boiler_ages.apply(
    lambda x: estimate_failure_year(x["first_failure"], dist_gb), axis=1
)
boiler_ages["third_failure"] = boiler_ages.apply(
    lambda x: estimate_failure_year(x["second_failure"], dist_gb), axis=1
)

# %%
boiler_ages["first_lifespan"] = boiler_ages["first_failure"] - boiler_ages["start_year"]
boiler_ages["second_lifespan"] = (
    boiler_ages["second_failure"] - boiler_ages["first_failure"]
)
boiler_ages["third_lifespan"] = (
    boiler_ages["third_failure"] - boiler_ages["second_failure"]
)

# %%
boiler_ages["first_lifespan"].describe()

# %%
boiler_ages["second_lifespan"].describe()

# %%
boiler_ages["third_lifespan"].describe()

# %%
heat_pump_ages["first_failure"] = heat_pump_ages.apply(
    lambda x: estimate_failure_year(x["start_year"], dist_hp), axis=1
)
heat_pump_ages["second_failure"] = heat_pump_ages.apply(
    lambda x: estimate_failure_year(x["first_failure"], dist_hp), axis=1
)
heat_pump_ages["third_failure"] = heat_pump_ages.apply(
    lambda x: estimate_failure_year(x["second_failure"], dist_hp), axis=1
)

# %%
heat_pump_ages["first_lifespan"] = (
    heat_pump_ages["first_failure"] - heat_pump_ages["start_year"]
)
heat_pump_ages["second_lifespan"] = (
    heat_pump_ages["second_failure"] - heat_pump_ages["first_failure"]
)
heat_pump_ages["third_lifespan"] = (
    heat_pump_ages["third_failure"] - heat_pump_ages["second_failure"]
)

# %%
heat_pump_ages["first_lifespan"].describe()

# %%
heat_pump_ages["second_lifespan"].describe()

# %%
heat_pump_ages["third_lifespan"].describe()

# %% [markdown]
# ### Modelling notification survival
#
# Let's apply our survival function to the notification data and build a model off of that.
# Basically, we'll predict how many boilers installed in a given year are still in use in 2023, and build a model from that.
#
# A survival function indicates the probability that the event of interest (e.g. failure) has not yet occurred by time t, in essence, it is the probability of surviving beyond time t. As such, we can use it as as scaling factor to estimate the number of boilers still in use.

# %%
# Gas boiler failure
gas_boiler_failure = weibull_min(c=2.2, loc=2, scale=13)

# %%
# apply survival function to boiler counts
notifications = notifications.assign(
    exist_2023=lambda df: df.apply(
        lambda x: int(gas_boiler_failure.sf(2023 - x["year"]) * x["count"]), axis=1
    )
)
notifications

# %%
f, ax = pyplot.subplots()

ax.bar(
    notifications["year"], notifications["count"], width=0.5, label="Boilers Installed"
)
ax.bar(
    notifications["year"] + 0.5,
    notifications["exist_2023"],
    width=0.5,
    label="Boilers Remaining, 2023",
)

ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.legend()

# %%
# model the age distribution
poly_exist_2023 = numpy.polynomial.Polynomial([2, 1]).fit(
    x=notifications["year"], y=notifications["exist_2023"], deg=2
)

# %%
poly_exist_2023

# %%
# Distribution based on new installs
# derived from polynomial fit to gas safe data
gas_boiler_year = (
    lambda x: 1036612.16083916
    + 276000.98901099 * (-335.83333333 + 0.16666667 * x)
    - 35931.77622378 * (-335.83333333 + 0.16666667 * x) ** 2
)

# %%
gas_boiler_new = (
    lambda x: 865042.97902098
    + 509250.95604396 * (-335.83333333 + 0.16666667 * x)
    - 74335.99000999 * (-335.83333333 + 0.16666667 * x) ** 2
)

# %%
counts_by_year = gas_boiler_year(numpy.arange(2003, 2023))
new_counts_by_year = gas_boiler_new(numpy.arange(2003, 2023))

# %%
f, ax = pyplot.subplots()

ax.plot(notifications["year"], notifications["exist_2023"])
ax.plot(numpy.arange(2003, 2023), new_counts_by_year)

# %%
# So now, with our new model, the counts of gas boilers actually go negative before 2007
f, ax = pyplot.subplots(figsize=(8, 4))

ax.bar(numpy.arange(2003, 2023), counts_by_year, width=0.5)
ax.bar(numpy.arange(2003, 2023) + 0.5, new_counts_by_year, width=0.5)

ax.set_xticks(numpy.arange(2003, 2023, 2))
ax.set_xlabel("Install Year")
ax.set_ylabel("Proportion of population")

ax.grid(axis="y")
ax.set_axisbelow(True)

# %%
counts_by_year = gas_boiler_year(numpy.arange(2007, 2023))
new_counts_by_year = gas_boiler_new(numpy.arange(2007, 2023))

prop_by_year = counts_by_year / sum(counts_by_year)
new_prop_by_year = new_counts_by_year / sum(new_counts_by_year)

# %%
f, ax = pyplot.subplots(figsize=(8, 4))

ax.bar(numpy.arange(2007, 2023), counts_by_year, width=0.5)
ax.bar(numpy.arange(2007, 2023) + 0.5, new_counts_by_year, width=0.5)

ax.set_xticks(numpy.arange(2007, 2023, 2))
ax.set_xlabel("Install Year")
ax.set_ylabel("Proportion of population")

ax.grid(axis="y")
ax.set_axisbelow(True)

# %%
# This does change the profile of the distribution somewhat
f, ax = pyplot.subplots(figsize=(8, 4))

ax.bar(numpy.arange(2007, 2023), prop_by_year, width=0.5)
ax.bar(numpy.arange(2007, 2023) + 0.5, new_prop_by_year, width=0.5)

ax.set_xticks(numpy.arange(2007, 2023, 2))
ax.set_xlabel("Install Year")
ax.set_ylabel("Proportion of population")

ax.grid(axis="y")
ax.set_axisbelow(True)
