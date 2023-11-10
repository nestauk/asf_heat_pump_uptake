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
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

# %% [markdown]
# ## Generalised cost of heating
#
# Generalised cost of heating is the present value cost of operating a heating system of technology $i$ throughout its lifespan, normalised for the production of 1 unit of heat per year, including non-monetary preferences.
#
# $$ \textrm{generalised cost of heating}_{i} = \frac{\displaystyle \sum_{t} \frac{\frac{IC_{it}}{CF_{i}} + \frac{MR_{it}}{CF_{i}} + \frac{FC_{it}}{CE_{i}}}{(1 + r)^{t}}}{\displaystyle \sum_{t} \frac{1}{(1 + r)^{t}}} + \gamma_{i} $$
#
# Where,
# ${IC}_{i}$ is technology $i$'s upfront investment cost.
# ${MR}_{i}$ is technology $i$'s maintenance-repair cost.
# ${FC}_{i}$ is technology $i$'s fuel cost.
# ${CF}_{i}$ is technology $i$'s capacity factor.
# ${CE}_{i}$ is technology $i$'s conversion efficiency.
# $r$ is a discount factor that expressed the realtive importance of future compared to present costs.
# $\gamma_{i}$ captures 'intangible' cost components and household preferences.
#
# Capacity factor describes the relationship between peak heat demand and annual heat demand. It is the ratio between average heat output and peak heat output, equivalent to: annual heat demand (kWh) / peak heat demand (kW) $\times$ 8760 hours.
#
# A typical capacity factor for space heating ranges from 17% for an intermittently occupied building (e.g. an office) to 24% for a continuously occupied building (e.g. a care home).
#
# As similar quantity used in MCS certification is 'full-load-equivalent hours' (FLEQ), which is the capacity factor $\times$ 8760 (hours in a year).
#
# Investment cost distributions (${IC}_{t}$) reflect a distribution of individual characteristics of the household or the dwelling, such as different installation costs, or preferences for specific brands.
#
# Meanwhile, the probability distributions associated with ${FC}_{t}$ and ${MR}_{t}$ correspond to the volatility of fuel prices and maintenance costs.
#
# In addition to this basic specification, policies can be imposed, as below:
#
# $$ \textrm{generalised cost of heating}_{i} = \frac{\displaystyle \sum_{t} \frac{\frac{IC_{it}(1 + T_{it})}{CF_{i}} + \frac{MR_{it}}{CF_{i}} + \frac{FC_{it} + FT_{it}}{CE_{i}} + FiT_{it}}{(1 + r)^{t}}}{\displaystyle \sum_{t} \frac{1}{(1 + r)^{t}} } + \gamma_{i} $$
#
# Where,
# $T_{i}$ is a technology specific subsidy/purchase tax on upfront investment costs (negative values are a subsidy, positive values a tax).
# ${FT}_{t}$ is a fuel tax (which can be specified for each fuel type and technology).
# ${FiT}_{t}$ is a technology-specific feed in-tariff which pays a pre-defined subsidy for each unit of produced heat.
#
# Refs
# https://www.sciencedirect.com/science/article/pii/S030142152100118X
# https://www.gov.scot/publications/economic-impact-decarbonising-heating-scotland/pages/10/
# https://energy.ec.europa.eu/system/files/2018-02/technical_analysis_residential_heat_0.pdf

# %%
# a standard deviation equivalent to 1/3 of the mean investment costs is assumed for all technologies.
# Mean upfront investment costs (incl. of installation costs, excl. of subsidies) for residential heating technologies
# in € per kW of thermal capacity.
IC_gas = norm(loc=504.87, scale=168)
IC_hp = norm(loc=837.50, scale=279)

# %%
# Operation and maintenance costs of residential heating systems, in € per kW of thermal capacity, and relative to
# investment costs/kW.
MR_gas = norm(loc=10.1, scale=2)
MR_hp = norm(loc=17.45, scale=3)

# %%
# Mean household prices for natural gas in €/kWh.
FC_gas = norm(loc=0.071, scale=0.01)
FC_hp = norm(loc=0.21, scale=0.01)

# %%
# Conversion efficiency - (output of useful energy per unit of fuel consumed on site)
CE_gas = 0.9
CE_hp = 2.5

# %%
# Capacity factor (MWh / kW)
CF_gas = 2.17
CF_hp = 1.71

# %%
denom_gas = norm(
    loc=sum([504.87 / 2.17, 10.1 / 2.17, 0.071 / 0.9]),
    scale=sum([168**2, 2**2, 0.01**2]) ** 0.5,
)
denom_hp = norm(
    loc=sum([837.50 / 1.71, 17.45 / 2.17, 0.21 / 2.5]),
    scale=sum([279**2, 3**2, 0.01**2]) ** 0.5,
)

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    numpy.arange(-500, 1500, 0.1),
    denom_gas.pdf(numpy.arange(-500, 1500, 0.1)),
    color="brown",
    label="gas boiler",
)
ax.plot(
    numpy.arange(-500, 1500, 0.1),
    denom_hp.pdf(numpy.arange(-500, 1500, 0.1)),
    color="green",
    label="heat pump",
)

ax.set_xlabel("Generalised cost of heating")
ax.legend()

# %%
binom = [
    0.5
    + 0.5
    * math.erf(
        (denom_hp.pdf(x) - denom_gas.pdf(x))
        / ((denom_gas.var() + denom_hp.var()) ** 0.5)
    )
    for x in numpy.arange(0.01, 1, 0.01)
]

# %%
plt.plot(binom, numpy.arange(0.01, 1, 0.01))
