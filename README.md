# Modelling Heat Pump Uptake to 2050

## Aim

The aim of this project is to model the pace of heat pump uptake in the UK, to help people identify how different policies and behaviours could alter the speed of uptake. We are concerned that the heat pump industry may face bottlenecks or surges in the lead up to a ban on new gas boilers, and want to help identify measures that could ease these.

We will do this by developing a microsimulation that allows us to vary the condition under which heat pumps are adopted and explore the affect that has on outcomes. We will work to prototype and incrementally improve the simulation approach, while building in visualisation and interactivity.

## Project Team

Project Lead - Daniel Lewis
Mission Sponsor - Andrew Sissons
Exec Sponsor - Elspeth Kirkman
Advisor (Data Science Practice) - Elizabeth Gallagher

## Approach

There are a number of challenges to address in designing a useful simulation model for this area of interest. We are taking an iterative approach, incrementally prototyping and testing models with new features.

The development naturally falls into two broad areas: development of a core model, and development of an intervention logic on top of that core model.

We have taken an object orientated approach to development as this aids the conceptualisation of the simulation process. This means that the model looks like an agent-based model, but without interaction among the agents.

The core model defines a set of household agents making heat source replacement decisions based on economic factors. Aspects of the household, heat source, fuel, upfront costs, and use of loan financing can be tailored to simulate an outcome.

## Development

If you'd like to modify or develop the source code you can clone it by first running:

`git clone git@github.com:nestauk/asf_heat_pump_uptake.git`

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
