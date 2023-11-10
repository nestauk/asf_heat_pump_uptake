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
import numpy
import matplotlib.pyplot as plt
from ipfn import ipfn
from scipy.stats import multinomial
from lmfit import Minimizer, Parameters, report_fit

# %%
rng = numpy.random.default_rng()

# %% [markdown]
# ### Key Variables
#
# The unit of analysis is the household.
#
# **Initial Assumptions**
# All household require a heating system.
# All Households have a choice between a 'fossil fuel heating system' and a 'low carbon heating system'.
# All households can be characterised by their behaviours.
# Households can be differentiated by attributes that affect heating cost and replacement decisions.
#
# In the first instance, it is important to limit the dimensionality of the simulated population. This will make it easier to prototype the boiler ban model as we'll be required to make fewer assumptions about how costs and replacement rates vary by household characteristics.
#
# Further, as the distribution of

# %% [markdown]
# ### One-way marginals (independent marginals)
#
# Efficient and parallelisable due to independence.
#
# But, doesn't capture statistical dependence across variables, so likely fails to capture the underlying structure of the real data.
#
# The key dimensions of relevance include:
# - main fuel.
# - housing stock.
# - tenure type.
# - housing age.
# - household income distribution.

# %%
# main fuel, gas, oil, solid, electric
fuel = multinomial.rvs(10000, p=[0.8803, 0.0324, 0.0043, 0.0830])
fuel_sample = numpy.repeat(["gas fired", "oil fired", "solid fuel", "electrical"], fuel)
rng.shuffle(fuel_sample)

# %%
# owner occupied, private rented, local authority, housing assoc.
tenure = multinomial.rvs(10000, p=[0.651, 0.180, 0.066, 0.103])
tenure_sample = numpy.repeat(
    ["owner occupied", "private rented", "local authority", "housing assoc."], tenure
)
rng.shuffle(tenure_sample)

# %%
synthetic_data = pandas.DataFrame(data={"fuel": fuel_sample, "tenure": tenure_sample})
synthetic_data

# %%

# %%
# http://freerangestats.info/blog/2019/11/03/re-creating-microdata

# %%
sd = pandas.crosstab(synthetic_data["tenure"], synthetic_data["fuel"])
sd

# %% [markdown]
# ## Boiler Age

# %% [markdown]
# ### National Energy Efficiency Database, 2014
#
# Public Use File.

# %%
need = pandas.read_csv("../data/need_public_use_file_2014.csv")

# %%
# 1= gas, 2=other
need["MAIN_HEAT_FUEL"].value_counts()

# %%
# 101: Detached House, 102: Semi-detached House, 103: End terrace, 104: Mid terrace, 105: Bungalow, 106: flat (inc. Maisonette)
need["PROP_TYPE"].value_counts()

# %%
# 101: Pre 1919/pre 1930
# 102: 1919-1944/1930-1949
# 103: 1945-1964/1950-1966
# 104: 1965-1982/1967-1982
# 105: 1983-1992/1983-1995
# 106: 1993-1999/1996 onwards
need["PROP_AGE"].value_counts()

# %%
need["BOILER_YEAR"].value_counts(dropna=False).sort_index()

# %%
# The 2006 and 2007 peak likely reflects the condensing boiler requirements.
f, ax = plt.subplots(figsize=(8, 4))

need["BOILER_YEAR"].value_counts().sort_index().plot(kind="bar", ax=ax)

# %%
need["boiler_age"] = (need["BOILER_YEAR"] - 2012).abs()

# %%
t = numpy.arange(0, 9)
x = numpy.array(need["boiler_age"].value_counts().sort_index())

# %%
x = numpy.array([1980, 1899, 1854, 1640, 1431, 1322, 1143, 894, 389])

# %%
# Create object for parameter storing
params_gompertz = Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_gompertz.add_many(
    ("N_0", numpy.log(x)[0], True, 0, None, None, None),
    ("N_max", numpy.log(x)[-1], True, 0, None, None, None),
    ("r_max", 0.62, True, None, None, None, None),
    ("t_lag", 5, True, 0, None, None, None),
)  # I see it in the graph


# %%
# Write down the objective function that we want to minimize, i.e., the residuals
def residuals_gompertz(params, t, data):
    """Model a logistic growth and subtract data"""
    # Get an ordered dictionary of parameter values
    v = params.valuesdict()
    # Logistic model
    model = v["N_0"] + (v["N_max"] - v["N_0"]) * numpy.exp(
        -numpy.exp(
            v["r_max"]
            * numpy.exp(1)
            * (v["t_lag"] - t)
            / ((v["N_max"] - v["N_0"]) * numpy.log(10))
            + 1
        )
    )
    # Return residuals
    return model - data


# %%
# Create a Minimizer object
minner = Minimizer(residuals_gompertz, params_gompertz, fcn_args=(t, numpy.log(x)))
# Perform the minimization
fit_gompertz = minner.minimize()

# %%
# Sumarize results
report_fit(fit_gompertz)

# %%
fit_gompertz.params

# %%
t_vec = numpy.arange(0, 15)
log_N_vec = numpy.ones(len(t_vec))
residual_smooth_gompertz = residuals_gompertz(fit_gompertz.params, t_vec, log_N_vec)

# %%
# The 2006 and 2007 peak likely reflects the condensing boiler requirements.
f, ax = plt.subplots(figsize=(8, 4))

need["boiler_age"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.plot(t_vec, numpy.exp(residual_smooth_gompertz + log_N_vec), color="r")
ax.set_xlim([-1, 15])

# %% [markdown]
# ## Multidimensional Iterative Proportional Fitting

# %%
# Uniform seed - just produces the distribution under independence - not of great additional benefit
m = numpy.ones((4, 4))

# main fuel, gas, oil, solid, electric
fuel = numpy.array([8803, 324, 43, 830])
# owner occupied, private rented, local authority, housing assoc.
tenure = numpy.array([6510, 1800, 660, 1030])

# %%
IPF = ipfn.ipfn(m, [tenure, fuel], [[0], [1]], convergence_rate=1e-6)

# %%
m = IPF.iteration()


# %%
def add_total(df):
    df.loc["Total"] = df.sum()
    return df


df = (
    pandas.DataFrame(m)
    .rename(
        columns={0: "gas", 1: "oil", 2: "solid", 3: "electric"},
        index={
            0: "owner occupied",
            1: "private rented",
            2: "local authority",
            3: "housing assoc.",
        },
    )
    .assign(total=lambda df: df.sum(axis=1))
    .pipe(add_total)
)
df

# %%
df.apply(lambda col: col / df[df.columns[:-1]].sum(axis=1)) * 100

# %%
observed = numpy.array(
    [
        [89.9, 4.2, 0.5, 5.5],
        [79.8, 2.4, 0.4, 17.3],
        [94.2, 0.1, 0.1, 5.6],
        [87.0, 0.6, 0.1, 12.1],
    ]
)

# %%
observed = pandas.DataFrame(observed).rename(
    columns={0: "gas", 1: "oil", 2: "solid", 3: "electric"},
    index={
        0: "owner occupied",
        1: "private rented",
        2: "local authority",
        3: "housing assoc.",
    },
)
observed

# %% [markdown]
# ### Bayesian Network (low dimensional (1,2,3- way) marginals)
#
# Generate synthetic records by sampling from Bayesian network.
#
# https://ermongroup.github.io/cs228-notes/inference/sampling/

# %%
sum(tenure)

# %% [markdown]
# ### Maximum Spanning Tree (MST)
#
# Using a Markov Random Field, carefully chosen 2-way marginals.
#
# Generate synthetic records by sampling from Markov Random Field.
#
# Chao-Lu alogrithm.

# %%

# %%
