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
import matplotlib.pyplot as plt

# %% [markdown]
# ## Epidemic Models
#
# Suppose that there are $N$ potential users of a new technology, and that each adopts the technology when they hear about it.
#
# At time $t$, $y(t)$ firms have adopted and ${N - y(t)}$ have not.
#
# Suppose further that information is transmitted from some central source, reaching $\alpha$% of the population each period.
#
# If $\alpha$ = 1, then the source contacts all $N$ potential users in the first period, and diffusion is instantaneous. If, on the other hand, $\alpha$< 1, then information spreads gradually and so, therefore, does usage of the new technology.
#
# A transmitter that contacts $\alpha$ % of the current population of non-users, ${N - y(t)}$, at time $t$ over the time interval $\Delta t$ increases awareness (or usage) by an amount $\Delta y(t) = \alpha {N - y(t)} \Delta t$, and, taking the limit as $\Delta t \rightarrow 0$ and solving for the time path of usage.
#
# $$y(t) = N\{1 - exp[- \alpha t]\}$$

# %%
# Number of Households
N = 28.1e6

# Alpha - awareness of heatpumps
alpha = 0.42

# Time
t = numpy.arange(0, 28, 1)

# %%
# estimate model
y = N * (1 - numpy.exp(-1 * alpha * t))

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(t, y)
ax.set_ylabel("Household Heat Pump Awareness")
ax.set_xlabel("Time")

# %% [markdown]
# The above is not an implausible model of how people become aware of heat pumps, but it definitely doesn't reflect uptake.
#
# Let's say instead that potential heat pump adopters need to be able to communicate directly with current users who have accumulated experience with the technology. This suggests that heat pump attitudes may follow a word of mouth information diffusion process in which the main source of information is previous users.
#
# Suppose that each existing user independently contacts a non-user with probability $\beta$.
#
# If there are $y(t)$ current users, then the probability that contact will be made with one of the ${N - y(t)}$ current non-users is $\beta y(t)$, meaning that usage will increase over the interval $\Delta t$ by an amount $\Delta y(t) = \beta y(t)\{N - y(t)\} \Delta t$.
#
# Assuming that there are $y(0) > 0$ initial users, taking the limit as $ \Delta t \rightarrow 0$ and solving for the time path of usage yields:
#
# $$y(t) = N\{1 + \phi exp[- \kappa t]\}^{-1}$$
#
# where:
# $\kappa \equiv \beta N$
# $\phi \equiv (N - y(0))/y(0)$
#

# %%
# Number of Households
N = 28.1e6

# Starting number of heat_pumps
y0 = 250_000

# phi - initial pool of heatpump owners
phi = (N - y0) / y0

# beta - diffusion rate
beta = 0.000000015

# Time
t = numpy.arange(0, 28, 1)

# %%
y = N * 1 / (1 + phi * numpy.exp(-1 * (beta * N) * t))

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(t, y)
ax.set_ylabel("Household Heat Pump Awareness (Word of Mouth)")
ax.set_xlabel("Time")

# %% [markdown]
# These two approaches can be mixed to give a 'mixed information source model. This is helpful as the word of mouth model cannot model uptake from technology inception as it relies on some amount of that technology already existing in the population.
#
# Since early adopting individuals or households have evidently chosen to use the technology despite not having had access to the experience of a previous user, it seems clear that they are somehow different from subsequent users. This suggests that a more satisfactory model should distinguish between (at least) two different types of agents.
#
# Over the time interval $\Delta t$, existing nonusers are subject to two sources of information, and the probability that one of them will be informed (or infected) is ${\alpha + \beta y(t)}$. This gives the model:
#
# $$y(t) = N\{1 - exp[-(\alpha / \sigma )t]\}\{1 + \psi exp[-(\beta / \sigma) t]\}^{-1}$$
#
# Where,
# $\sigma \equiv \alpha / (\alpha + \kappa)$ with $\kappa$ controlling information in the system.
# if $\kappa$ = 0, then no word of mouth transmission occurs and $\sigma$ = 1,
# while if $\alpha$ = 0 then the common source does not broadcast and $\alpha$ = 0.
#
# Note that when $\alpha$ = 0, $y(t)$ = 0 for all $t$ since no common source of information exists to create the initial user base that is needed to start a word of mouth process.
#
# When $\sigma$ is “small” then the time path of $y(t)$ will resemble a logistic curve with an inflection point at $(N/2){(1-2 \sigma)/(1- \sigma)} < N/2$. As $\sigma$ increases, the inflection point falls and the logistic curve becomes increasingly asymmetric.
#
# Not clear what psi is... it should be some kind of indicator of the population available to word of mouth though.

# %%
# Number of Households
N = 28.1e6

# Alpha - awareness of heatpumps
alpha = 0.05

# beta - diffusion rate
beta = 0.0000000015

# kappa
kappa = 0.1

# sigma
sigma = alpha / (alpha + kappa)

# Time
t = numpy.arange(0, 28, 1)

# %%
y = []
# not sure about this...
for time in t:
    central_bit = N * (1 - numpy.exp(-1 * (alpha / sigma) * time))
    word_of_mouth_bit = 1 / (
        1
        + ((N - central_bit) / central_bit)
        * numpy.exp(-1 * ((beta * N) / sigma) * time)
    )
    y.append(central_bit * word_of_mouth_bit)

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(t, y)
ax.set_ylabel("Household Heat Pump Awareness (Word of Mouth)")
ax.set_xlabel("Time")

# %% [markdown]
# In any case, these models all make more sense in the diffusion of information rather than of artefacts, like heat pumps. The relatively even flow of information between individuals which they are built on is plausible only when applied to homophilic populations. When populations are heterophilic, differences between individuals can impede the process of communication or, more likely, the process of persuasion.
#
# Suppose that there are two populations, $N_{1}$ and $N_{2}$, which do not interact with each other. Each has an initial number of users, $y_{1}(0)$ and $y_{2}(0)$, who initiate word of mouth diffusion processes with speeds $\beta_{1}$ and $\beta_{2}$ respectively. Following exactly the same argument as earlier, the increase in the total number of users, $y(t) \equiv y1(t) + y2(t)$, over the interval $\Delta t$ is:
#
# $$\Delta y(t) = \{\beta_{1}y_{1}(t)\{N_{1} - y_{1}(t)\} + \beta_{2}y_{2}(t)\{N_{2} - y_{2}(t)\}\} \Delta t$$
#
# It is relatively easy to extend this to the case where the two groups interact. Suppose, for example, that users in population 1 contact non-users in population 2 at a rate $\eta_{12}$ while users in 2 contact non-users in 1 at a rate $\eta_{21}$.
#
# $$\Delta y(t) = {[\beta_{1} y_{1}(t) + \eta_{12}y_{2}(t)]{N_{1} - y_{1}(t)} + [\beta_{2}y_{2}(t) + \eta_{21}y_{1}(t)]{N_{2}y_{2}(t)}} \Delta t$$
#
# which is very similar to the Lotka-Volterra model of competitive exclusion.
#
# One of the big problems with the epidemic model is that it takes $N$ and $\beta$ as fixed, and the two population model that we have just discussed is useful because it is an easy way to get round these drawbacks. In particular, it can be used to mimic a process in which $\beta$ declines over time. There are any number of reasons why this might happen. One obvious possibility is that users become increasingly resistant to word of mouth communication (i.e. resistance to the disease increases and infection rates fall off); another is that late adoptersmay simply be less able to understand the new technology than early adopters.
#
# It is worth emphasising that one rarely encounters symmetric S-curves in the actual diffusion of new technology. In almost all cases, the later stages of diffusion occur much more slowly than would be predicted by a symmetric S-curve.

# %% [markdown]
# ## Lotka - Volterra Models
#
# Note that this style of predator-prey-like models are what powers the European FTT:Heat model.
#
# https://dspace.mit.edu/bitstream/handle/1721.1/2635/SWP392936119517.pdf
#
# The term *competition* is frequently used in the context of innovation economics. The interaction between technologies is, however, often not one of competition in the strict sense of the word, for there are many cases where technologies interact in a relationship that is not confrontational. The concept of growth rate offers itself as a more general way of classifying the process of interaction among technologies, so that interaction can be manifested in the concept of the reciprocal effect that one technology has one another's growth rate.
#
# Three possible modes of interaction can exist: 1, *pure competition* where both technologies inhibit the other's growth rate, 2,  *symbiosis* where both technologies enhance the other's growth rate, and 3, *predator-prey interaction* where one technology enhances the other's growth rate but the second inhibits the growth rate of the first.
#
# Two technologies $N$ and $M$, e.g. heat pumps and gas boilers can be understood as interacting with one another according to the following equations:
#
# $$ \frac{dN}{dt} = a_{n}N - b_{n}N^{2} \pm c_{nm}NM $$
#
# and,
#
# $$ \frac{dM}{dt} = a_{m}M - b_{m}M^{2} \pm c_{mn}MN $$
#
# Two positive signs of the coefficients $c$ indicate symbiotic interaction, two negative signs indicate pure competition whereas one positive and one negative sign indicates predator-prey interaction.
#
# Let's look at the following case: $a_{n}$ = 0.1, $a_{m}$ = 0.15, $b_{n}$ = $b_{m}$ = 0.01, $c_{nm}$ = 0.02 and $c_{mn}$ = 0.01. Let the initial conditions be $N(0)$ = 0.01 and $M(0)$=5, i.e. $N$ is an emerging technology which is attacking a mature technology ($M$).

# %%
time = range(0, 151)

a_n = 0.1
a_m = 0.15
b_n = b_m = 0.01
c_nm = 0.02
c_mn = 0.01

N = [0.01]
M = [5]

for t in time:
    N.append((a_n * N[t] - b_n * N[t] ** 2 + c_nm * N[t] * M[t]) + N[t])
    M.append((a_m * M[t] - b_m * M[t] ** 2 - c_mn * M[t] * N[t]) + M[t])

# %%
f, ax = plt.subplots(figsize=(8, 4))

ax.plot(N, label="Emerging Tech")
ax.plot(M, label="Mature Tech")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("N(t), M(t)")

# %% [markdown]
# ## Bass Diffusion Model
#
# The basic premise of the model is that adopters can be classified as innovators or as imitators, and the speed and timing of adoption depends on their degree of innovation and the degree of imitation among adopters. The intuition is that the “innovators”, will adopt the product regardless of who has bought it before, and then convey the information through interpersonal communication (word-of-mouth) to the “imitators”, who adopt the product later.
#
# The basic model is a differential equation that looks like:
#
# $$ \frac{dF}{dt} = p(1 - F) + q(1 - F)F = (1 - F)(p + qF)$$
#
# where,
# $p$ = the coefficient of innovation, external influence or advertising effect. Represents the probability that an innovator will adopt at time t.
# $q$ = the coefficient of imitation, internal influence or word-of-mouth effect.
# $F$ = the installed base fraction.
#
# The installed base fraction $F(t)$ is then given by:
#
# $$F(t) =  \frac{ 1 - {\rm e}^{-(p + q)t}}{1 + \frac{q}{p}{\rm e}^{-(p + q)t}}$$
#
# The number of innovators at time $t$ is $mp(1 - F(t))$, while the number of imitators is $mq(1-F(t))F(t)$

# %%
time = numpy.linspace(0, 30, 1000)
p = 0.03
q = 0.38
m = 28.1e6

adopters = []

for t in time:
    adopters.append(
        (1 - numpy.exp(-1.0 * (p + q) * t))
        / (1 + (q / p) * numpy.exp(-1.0 * (p + q) * t))
    )

innovators = numpy.array([m * p * (1 - adopter) for adopter in adopters])
imitators = numpy.array([m * q * (1 - adopter) * adopter for adopter in adopters])

# %%
f, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(time, m * numpy.array(adopters))
ax1.set_xlabel("Time, years")
ax1.set_ylabel("Cumulative # of low carbon heat sources")

ax2.plot(time, innovators + imitators, label="New Adopters")
ax2.plot(time, innovators, label="Innovators")
ax2.plot(time, imitators, label="Imitators")
ax2.set_xlabel("Time, years")
ax2.set_ylabel("Number of low carbon heat sources")
ax2.legend()

f.suptitle("Bass Diffusion Model, p=0.03, q=0.38")


# %%
## Generalised

# %%
# https://github.com/cran/DIMORA/blob/master/R/GBM.R

# a = starting time of the shock
# b = memory of the effect (typically negative, suggesting an exponentially decaying behavior)
# c = intensity of the shock (maybe either positive or negative)


# exponential shock
def intx(t, a1, b1, c1):
    return t + c1 * (1 / b1) * (numpy.exp(b1 * (t - a1)) - 1) * (t >= a1)


def xt(t, a1, b1, c1):
    return 1 + (c1 * numpy.exp(b1 * (t - a1))) * (t >= a1)


def ff(t, m, p, q, a1, b1, c1):
    return (
        m
        * (1 - numpy.exp(-(p + q) * intx(t, a1, b1, c1)))
        / (1 + (q / p) * numpy.exp(-(p + q) * intx(t, a1, b1, c1)))
    )


def zprime(t, m, p, q, a1, b1, c1):
    return (
        m
        * (p + q * (ff(t, m, p, q, a1, b1, c1) / m))
        * (1 - (ff(t, m, p, q, a1, b1, c1) / m))
        * xt(t, a1, b1, c1)
    )


# %%
adopt = [zprime(t, m, p, 0.2, 15, -0.0001, 0) for t in time]
adopt1 = [zprime(t, m, p, 0.2, 15, -0.0001, 1.2) for t in time]

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(time, adopt)
ax.plot(time, adopt1)

# %% [markdown]
# ## Probit Models
#
# https://repec.cepr.org/repec/cpr/ceprdp/DP2146.pdf
#
# https://www.aceee.org/files/proceedings/2001/data/papers/SS01_Panel1_Paper45.pdf
