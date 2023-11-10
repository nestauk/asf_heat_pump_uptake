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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy
import pandas

# %%
initial_gas_boilers = 23_000_000

assumptions = {
    "year_full_ban_date_announced": 2024,
    "todays_date": datetime.today().strftime("%Y-%m-%d"),
    "year_new_boilers_banned": 2035,
    "year_all_boilers_banned": 2050,
    "initial_gas_boilers": initial_gas_boilers,
    "boiler_annual_replacement_rate": (1_750_000 / initial_gas_boilers),
    "annual_growth_households_needing_heating": 0.0071,
    "pre_ban_boiler_replacement_rate": 0.98,
    "first_time_hp_installs_per_installer_per_year": 22,
    "replacement_hp_installs_per_installer_per_year": 40,
    "hp_annual_replacement_rate": (initial_gas_boilers / 15 / initial_gas_boilers),
    "initial_heat_pumps": 250_000,
    "voluntary_replacement_rate_at_ban_announcement": 0.01,
    "voluntary_replacement_rate_year_on_year_increase": 0.01,
    "hp_lifespan": 15,
}


# %%
def get_total_households_that_require_heating(
    year, base_households, growth_rate, base_year=2023
):
    """Calculate compound growth in houses needing heating."""
    return (
        base_households * (1 + growth_rate) ** (year - base_year)
        if year >= base_year
        else base_households
    )


def get_gas_boilers_added_from_new_homes(year, ban_year, base_households, growth_rate):
    """Calculate number of gas boilers added from new homes, subject to boiler ban."""
    return (
        0.0
        if year >= ban_year
        else (
            get_total_households_that_require_heating(
                year, base_households, growth_rate
            )
            - get_total_households_that_require_heating(
                year - 1, base_households, growth_rate
            )
        )
    )


def get_heat_pumps_added_from_new_homes(year, ban_year, base_households, growth_rate):
    """Calculate number of heat pumps added from new homes, subject to boiler ban."""
    return (
        0.0
        if year < ban_year
        else (
            get_total_households_that_require_heating(
                year, base_households, growth_rate
            )
            - get_total_households_that_require_heating(
                year - 1, base_households, growth_rate
            )
        )
    )


# %%
# End of life functions
def get_retirement_rate_multiplier(year, ban_year):
    """Years since ban on new boilers multiplier."""
    return 1 if year < ban_year else (year - ban_year) + 2


def get_retirement_rate(year, ban_year, boiler_annual_replacement_rate):
    """Retirement rate each year as remaining boilers get older on average."""
    return (
        1
        if (
            get_retirement_rate_multiplier(year, ban_year)
            * boiler_annual_replacement_rate
        )
        > 1
        else get_retirement_rate_multiplier(year, ban_year)
        * boiler_annual_replacement_rate
    )


def get_voluntary_switch_rate(
    year,
    total_ban_announce_year,
    voluntary_replacement_rate_at_ban_announcement,
    voluntary_replacement_rate_year_on_year_increase,
):
    """Proportion of households voluntarily switching."""
    if year < total_ban_announce_year:
        return 0
    elif year == total_ban_announce_year:
        return voluntary_replacement_rate_at_ban_announcement
    else:
        return (1 + voluntary_replacement_rate_year_on_year_increase) ** (
            year - total_ban_announce_year
        ) / 100


# %%
def get_gas_boilers_retired_without_replacement(
    year,
    ban_year,
    total_ban_year,
    pre_ban_boiler_replacement_rate,
    boiler_annual_replacement_rate,
    current_gas_boilers,
):
    """Get the number of gas boilers retired without being replaced by a new gas boiler."""
    if year >= total_ban_year:
        return 0  # i.e. you cannot own or operate a gas boiler.
    elif year < ban_year:
        # This is the people who might need to replace their boiler who are also future thinking
        return (
            (1 - pre_ban_boiler_replacement_rate)
            * current_gas_boilers
            * boiler_annual_replacement_rate
        )
    elif year >= ban_year:
        # get replacements from last years retirement rate and boiler numbers.
        forced_replacements = (
            get_retirement_rate(year - 1, ban_year, boiler_annual_replacement_rate)
            * current_gas_boilers
        )
        voluntary_replacements = (
            (1 - pre_ban_boiler_replacement_rate)
            * current_gas_boilers
            * boiler_annual_replacement_rate
        )
        return forced_replacements + voluntary_replacements
    else:
        return None


# %%
def get_gas_boilers_switched_off_due_to_full_ban(
    year,
    total_ban_year,
    total_ban_announced_year,
    current_gas_boilers,
    voluntary_replacement_rate_at_ban_announcement,
    voluntary_replacement_rate_year_on_year_increase,
):
    """Get the voluntary gas boiler switch-offs."""
    # all boilers are switched off in the ban year, else none are.
    mass_switch_off = current_gas_boilers if year == total_ban_year else 0
    # knowing a ban exists, some people choose to voluntarily switch ahead of full ban year
    voluntary_switch_off = (
        current_gas_boilers
        * get_voluntary_switch_rate(
            year,
            total_ban_announced_year,
            voluntary_replacement_rate_at_ban_announcement,
            voluntary_replacement_rate_year_on_year_increase,
        )
        if (year > total_ban_announced_year) & (year <= total_ban_year)
        else 0
    )

    return mass_switch_off + voluntary_switch_off


# %%
def get_heat_pumps_replaced(
    year,
    hp_lifespan,
    current_heat_pumps_list,
    heat_pump_replacement_rate,
    initial_heat_pumps,
    base_year=2023,
):
    if year - base_year > hp_lifespan:
        return current_heat_pumps_list[-hp_lifespan] * heat_pump_replacement_rate
    else:
        return heat_pump_replacement_rate * initial_heat_pumps


# %%
# current_gas_boilers is a free variable.
# This means it currently only works in a sequence, not for an arbitary year.
current_gas_boilers = assumptions["initial_gas_boilers"]
current_heat_pumps = assumptions["initial_heat_pumps"]

# data collector variables
gas_boiler_count = []
new_gas_count = []
new_heat_pump_count = []
retired_gas_count = []
volunteer_retired_gas_count = []
heat_pumps_count = []
heat_pump_conversions = []
heat_pump_replacements = []
required_installer_count = []

for current_year in range(2023, 2051):
    # calculate gas boilers added from new homes
    new_gas = get_gas_boilers_added_from_new_homes(
        current_year,
        assumptions["year_new_boilers_banned"],
        assumptions["initial_gas_boilers"],  # stand in for number of hhs
        assumptions["annual_growth_households_needing_heating"],
    )
    # Calculate the gas boilers retired without replacement (e.g. flow to heat pumps)
    retired_gas = get_gas_boilers_retired_without_replacement(
        current_year,
        assumptions["year_new_boilers_banned"],
        assumptions["year_all_boilers_banned"],
        assumptions["pre_ban_boiler_replacement_rate"],
        assumptions["boiler_annual_replacement_rate"],
        current_gas_boilers,
    )
    # Calculate the flow to heat pumps caused by people getting ahead of a full ban
    volunteer_retired_gas = get_gas_boilers_switched_off_due_to_full_ban(
        current_year,
        assumptions["year_all_boilers_banned"],
        assumptions["year_full_ban_date_announced"],
        current_gas_boilers,
        assumptions["voluntary_replacement_rate_at_ban_announcement"],
        assumptions["voluntary_replacement_rate_year_on_year_increase"],
    )
    # calculate new build heat pumps
    new_heat_pumps = get_heat_pumps_added_from_new_homes(
        current_year,
        assumptions["year_new_boilers_banned"],
        assumptions["initial_gas_boilers"],  # stand in for number of hhs
        assumptions["annual_growth_households_needing_heating"],
    )
    # calculate required heat pump replacements (e.g. 2nd heat pump)
    replacement_heat_pumps = get_heat_pumps_replaced(
        current_year,
        assumptions["hp_lifespan"],
        heat_pumps_count,
        assumptions["hp_annual_replacement_rate"],
        assumptions["initial_heat_pumps"],
    )

    # update current gas boilers
    current_gas_boilers += new_gas - retired_gas - volunteer_retired_gas
    # update current heat pumps
    current_heat_pumps += retired_gas + volunteer_retired_gas + new_heat_pumps

    # compute required installers
    required_installers = (
        (retired_gas + volunteer_retired_gas + new_heat_pumps)
        / assumptions["first_time_hp_installs_per_installer_per_year"]
    ) + (
        replacement_heat_pumps
        / assumptions["replacement_hp_installs_per_installer_per_year"]
    )

    # capture all data
    gas_boiler_count.append(current_gas_boilers)
    heat_pumps_count.append(current_heat_pumps)
    new_gas_count.append(new_gas)
    new_heat_pump_count.append(new_heat_pumps)
    retired_gas_count.append(retired_gas)
    volunteer_retired_gas_count.append(volunteer_retired_gas)
    heat_pump_conversions.append(retired_gas + volunteer_retired_gas)
    heat_pump_replacements.append(replacement_heat_pumps)
    required_installer_count.append(required_installers)

# %%
# Combine everything into pandas dataframe
model_data = pandas.DataFrame(
    data={
        "year": range(2023, 2051),
        "gas boiler count": gas_boiler_count,
        "heat pump count": heat_pumps_count,
        "new build gas boiler count": new_gas_count,
        "new build heat pump count": new_heat_pump_count,
        "gas boilers retired without replacement": retired_gas_count,
        "gas boilers voluntarily retired": volunteer_retired_gas_count,
        "heat pump conversions": heat_pump_conversions,
        "heat pump replacements": heat_pump_replacements,
        "installers required": required_installer_count,
    }
)

model_data

# %%
f, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    range(2023, 2051),
    model_data["gas boiler count"],
    color="dodgerblue",
    label="Gas Boiler Count",
)
ax.plot(
    range(2023, 2051),
    model_data["heat pump count"],
    color="darkblue",
    label="Heat Pump Count",
)
ax.plot(
    range(2023, 2051),
    model_data["new build gas boiler count"].cumsum(),
    color="indianred",
    label="New Build Gas Boilers Cumulative",
)
ax.plot(
    range(2023, 2051),
    model_data["gas boilers retired without replacement"].cumsum(),
    color="forestgreen",
    label="Retired Gas Cumulative",
)
ax.plot(
    range(2023, 2051),
    model_data["gas boilers voluntarily retired"].cumsum(),
    color="gold",
    label="Volunteer Retired Gas Cumulative",
)
ax.plot(
    range(2023, 2051),
    model_data["new build heat pump count"].cumsum(),
    color="purple",
    label="New Build Heat Pumps Cumulative",
)
ax.legend(loc=6)
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Number of Households")

# %%
f, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    range(2023, 2051),
    model_data["gas boiler count"],
    color="dodgerblue",
    label="Gas Boiler Count",
)
ax.plot(
    range(2023, 2051),
    model_data["heat pump count"],
    color="darkblue",
    label="Heat Pump Count",
)
ax.plot(
    range(2023, 2051),
    model_data["new build gas boiler count"],
    color="indianred",
    label="New Build Gas Boilers",
)
ax.plot(
    range(2023, 2051),
    model_data["gas boilers retired without replacement"],
    color="forestgreen",
    label="Retired Gas",
)
ax.plot(
    range(2023, 2051),
    model_data["gas boilers voluntarily retired"],
    color="gold",
    label="Volunteer Retired Gas",
)
ax.plot(
    range(2023, 2051),
    model_data["new build heat pump count"],
    color="purple",
    label="New Build Heat Pumps",
)
ax.legend(loc=6)
ax.grid()
ax.set_xlabel("Year")
ax.set_ylabel("Number of Households")

# %%
f, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    range(2023, 2051), model_data["installers required"], label="Required Installers"
)
ax.legend()
ax.grid()
ax.set_ylabel("Installer Count")
