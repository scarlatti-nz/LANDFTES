# Import modules
from ast import Str
from functools import lru_cache
import dash
from dash import dcc, html, ctx, callback
from dash.dependencies import Output, State, Input
from dash.exceptions import PreventUpdate
from flask import Flask
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from pandas._libs.missing import NAType
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from math import floor, log10
import itertools
import scipy as sp
from scipy import optimize
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import base64
from email import encoders
import smtplib
from email_params import email_params
import plotly

dash.register_page(__name__, path="/", name="Home", title="Home")

# Read data files
# Model worker database
df = pd.read_csv("data/worker_database_8_1_24.csv")

# Seasonal complements database
complement = pd.read_csv("data/seasonal_complementarities.csv")
regional_exclusions = pd.read_csv("data/regional_suitability_master_updated_may_2024.csv")
COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS
# transform colors into transparent colors by translating rgb strings into rgba strings with alpha=0.2
COLORS_TRANSPARENT = [
    c.split("(")[0] + "a(" + c.split("(")[1].split(")")[0] + ",0.2)" + c.split(")")[1]
    for c in COLORS
]

# Define model elements
# Roles, if not specified in database default to "worker" only
if "role" not in df:
    df["role"] = "worker"
roles = df.role.unique()
df["task"] = df["task"].fillna("No task data")
df["management_category"] = df["management_category"].fillna(4)
# Land uses
land_uses = df.land_use.unique()
# Months of year
months_of_year = df.month_of_year.unique()
# Years since establishment
years_since_start = df.year_since_start.unique()
stages = ["start", "growing", "established"]  # df.stage.unique()
df["hours_ha"] = df["hours_ha"].fillna(0)


# Image files
olw_logo = "assets/olwlogofull.svg"
scarlatti_logo = "assets/White Scarlatti secondary logo no tagline.svg"

# TO ADD: Add in 30-year average to stages and calculate details

# Define defaults
land_area_ha = 1
yearly_hours_x_fte = 1840

# Build components
# Dashboard title component --NOT CURRENTLY IN USE--
header_component = html.H1(
    "Our Land and Water",
    style={
        "color": "white",
        "text-align": "center",
        "font": "Helvetica",
        "backgroundColor": "#41a0b0",
    },
)
# Dashboard subtitle component --NOT CURRENTLY IN USE--
sub_header_component = html.H1(
    "Land and Labour Use Dashboard",
    style={
        "color": "white",
        "text-align": "center",
        "font": "Helvetica",
        "backgroundColor": "#41a0b0",
    },
)
# Filter pan header component
filter_pane_header_component = html.H1(
    "Inputs",
    style={
        "color": "white",
        "display": "inline-block",
        "width": "100%",
        "text-align": "center",
        "font": "Helvetica",
        "backgroundColor": "#41a0b0",
        "font-size": "24px",
        "margin-bottom": "2rem",
        "vertical-align": "bottom",
    },
)

# Set default selections for filters
selected_land_uses = ["Blueberry"]
selected_roles = ["worker"]
selected_stages = ["established"]

# Check selected land uses are a subset of the set of all land uses
assert set(selected_land_uses).issubset(set(land_uses))


def subset_to_selections_yearly(df, lus, roles, stages):
    df = df.loc[df["land_use"].isin(lus), :]
    df = df.loc[df["role"].isin(roles), :]
    df = df.loc[df["stage"].isin(stages), :]
    df = df[pd.isna(df["default_planting_month"]) | (df["default_planting_month"] == 1)]
    df = df[pd.isna(df["default_harvest_month"]) | (df["default_harvest_month"] == 1)]
    # sum hours for selected worker categories
    collapsed_to_lu_year = (
        # df.drop("month_of_year", axis=1)
        df.groupby(
            [
                "month_of_year",
                "role",
                "management_category",
                "land_use",
                "stage",
                "year_since_start",
            ]
        ).sum(numeric_only=True)
    )
    collapsed_to_lu_year = (
        collapsed_to_lu_year.groupby(
            ["year_since_start", "management_category", "role", "land_use"]
        )
        .sum()
        .reset_index()
    )
    # if any numbers between 0 and 30 don't exist in the year_since_start column, add them with 0 hours
    template = collapsed_to_lu_year.loc[0]
    template["hours_ha"] = 0
    rows_to_add = []
    for i in range(30):
        if i not in collapsed_to_lu_year["year_since_start"].values:
            template["year_since_start"] = i
            rows_to_add.append(template.to_list())

    rows = pd.DataFrame(rows_to_add, columns=collapsed_to_lu_year.columns)
    collapsed_to_lu_year = pd.concat([collapsed_to_lu_year, rows])
    collapsed_to_lu_year = collapsed_to_lu_year.sort_values(by="year_since_start")

    return collapsed_to_lu_year


def subset_to_selections_and_collapse(df, lus, roles, stages):
    df = df.loc[df["land_use"].isin(lus), :]
    df = df.loc[df["role"].isin(roles), :]
    df = df.loc[df["stage"].isin(stages), :]
    # sum hours for selected worker categories

    collapsed_to_month_lu_year = (
        # df.drop("role", axis=1)
        df.groupby(
            [
                "month_of_year",
                "task",
                "management_category",
                "harvest_month",
                "planting_month",
                "role",
                "land_use",
                "stage",
                "year_since_start",
            ],
            dropna=False,
        ).sum(numeric_only=True)
    )
    # average hours by year_since_start
    collapsed_to_month_lu = (
        collapsed_to_month_lu_year.groupby(
            [
                "month_of_year",
                "task",
                "management_category",
                "harvest_month",
                "planting_month",
                "role",
                "land_use",
            ],
            dropna=False,
        )
        .mean()
        .reset_index()
    )
    return collapsed_to_month_lu


collapsed_to_month_lu = subset_to_selections_and_collapse(
    df, land_uses, selected_roles, selected_stages
)

hours_x_month = go.FigureWidget()
ftes_x_year = {lu: go.Figure() for lu in selected_land_uses}
wages_x_year = {lu: go.Figure() for lu in selected_land_uses}


def merge_tasks(tmp):
    harvest_months = pd.unique(tmp["harvest_month"].dropna()).tolist()
    planting_months = pd.unique(tmp["planting_month"].dropna()).tolist()
    if len(harvest_months) > 1:
        tasklist = ["" for i in range(12)]
        for h in harvest_months:
            new_tasklist = tmp.loc[tmp["harvest_month"] == h, "task"]
            for i in range(12):
                if new_tasklist.iloc[i] not in tasklist[i]:
                    if tasklist[i] in new_tasklist.iloc[i]:
                        tasklist[i] = new_tasklist.iloc[i]
                    else:
                        tasklist[i] = tasklist[i] + ", " + new_tasklist.iloc[i]
        for h in harvest_months:
            try:
                tmp.loc[tmp["harvest_month"] == h, "task"] = tasklist
            except:
                print("couldn't assign task list")

    elif len(planting_months) > 1:
        tasklist = ["" for i in range(12)]
        for h in planting_months:
            new_tasklist = tmp.loc[tmp["planting_month"] == h, "task"]
            for i in range(12):
                if new_tasklist.iloc[i] not in tasklist[i]:
                    if tasklist[i] in new_tasklist.iloc[i]:
                        tasklist[i] = new_tasklist.iloc[i]
                    else:
                        tasklist[i] = tasklist[i] + ", " + new_tasklist.iloc[i]
        tasklist = [
            t.replace("No task data, ", "").replace(", No task data", "")
            for t in tasklist
        ]
        for h in planting_months:
            try:
                tmp.loc[tmp["planting_month"] == h, "task"] = tasklist
            except:
                print("couldn't assign task list")
    elif len(tmp) > 12:
        print(
            f"WARNING: NONUNIQUE TASK-STAGE COMBINATION FOR LAND USE: {tmp['land_use'].iloc[0]}"
        )
        # Very hacky fix, shouldn't ever have to run, hence above warning.
        tmp.sort_values(by=["task", "month_of_year"], inplace=True)
        tmp["itercol"] = 0
        iters = int(len(tmp) / 12)
        tasklist = ["" for i in range(12)]

        for i in range(iters):
            new_tasklist = tmp.iloc[i * 12 : (i + 1) * 12]["task"]
            tmp["itercol"] += [
                i if j in range(i * 12, (i + 1) * 12) else 0 for j in range(len(tmp))
            ]
            for j in range(12):
                if new_tasklist.iloc[j] not in tasklist[j]:
                    if tasklist[j] in new_tasklist.iloc[j]:
                        tasklist[j] = new_tasklist.iloc[j]
                    else:
                        tasklist[j] = tasklist[j] + ", " + new_tasklist.iloc[j]
        tasklist = [
            t.replace("No task data, ", "").replace(", No task data", "")
            for t in tasklist
        ]
        for i in range(iters):
            tmp.loc[tmp["itercol"] == i, "task"] = tasklist
        tmp.drop("itercol", axis=1, inplace=True)
    return tmp


# FIX THIS


def create_hours_by_month_for_one_land_use(
    df, lu, land_area_ha, role, start_month=None, end_month=None
):
    tmp = df.loc[(df["land_use"] == lu) & (df["role"] == role)].copy()
    if not tmp.planting_month.isna().all():
        if not (start_month and end_month):
            try:
                default = tmp.loc[
                    tmp["default_planting_month"] == 1, "planting_month"
                ].iloc[0]
                start_month = default
                end_month = default
            except:
                start_month = 1
                end_month = 12

        tmp = (
            tmp.loc[
                (tmp["planting_month"] >= start_month)
                & (tmp["planting_month"] <= end_month)
            ]
            if end_month >= start_month
            else tmp.loc[
                (tmp["planting_month"] >= start_month)
                | (tmp["planting_month"] <= end_month)
            ]
        )
    elif not tmp.harvest_month.isna().all():
        if not (start_month and end_month):
            try:
                default = tmp.loc[
                    tmp["default_harvest_month"] == 1, "harvest_month"
                ].iloc[0]
                start_month = default
                end_month = default
            except:
                start_month = 1
                end_month = 12
        tmp = (
            tmp.loc[
                (tmp["harvest_month"] >= start_month)
                & (tmp["harvest_month"] <= end_month)
            ]
            if end_month >= start_month
            else tmp.loc[
                (tmp["harvest_month"] >= start_month)
                | (tmp["harvest_month"] <= end_month)
            ]
        )

    tmp = merge_tasks(tmp)
    tmp = (
        tmp.groupby(
            [
                c
                for c in tmp.columns
                if c
                not in [
                    "planting_month",
                    "harvest_month",
                    "default_harvest_month",
                    "default_planting_month",
                    "hours_ha",
                ]
            ]
        )
        .mean()
        .reset_index()
    )

    val = pd.to_datetime(tmp["month_of_year"] * 100 + 20220001, format="%Y%m%d")
    val = val.dt.month_name().str.slice(stop=3)
    tmp["month_short_name"] = val
    tmp["hours_total"] = tmp["hours_ha"] * land_area_ha
    # print(tmp)
    return tmp


def calculate_yearly_ftes(df, lu, land_area_ha, role):
    tmp = create_hours_by_month_for_one_land_use(df, lu, land_area_ha, role)
    hours_total = sum(tmp["hours_total"])
    return hours_total / yearly_hours_x_fte


def calculate_yearly_wage_cost(df, lu, land_area_ha, role, worker_wage, manager_wage):
    ftes = calculate_yearly_ftes(df, lu, land_area_ha, role)
    wage = worker_wage if role == "worker" else manager_wage
    total_wage = ftes * yearly_hours_x_fte * wage
    # total_wage = "$" + "{:.2f}".format(round(total_wage, 2))
    return total_wage


for lu in selected_land_uses:
    tmp = create_hours_by_month_for_one_land_use(
        collapsed_to_month_lu, lu, land_area_ha, "worker"
    )
    hours_x_month.add_trace(go.Scatter(x=tmp.month_short_name, y=tmp.hours_total))

lu_yearly_wage = {
    lu: {
        role: calculate_yearly_wage_cost(
            collapsed_to_month_lu, lu, land_area_ha, role, 25, 35
        )
        for role in selected_roles
    }
    for lu in land_uses
}

lu_yearly_fte = {
    lu: {
        role: calculate_yearly_ftes(collapsed_to_month_lu, lu, land_area_ha, role)
        for role in selected_roles
    }
    for lu in land_uses
}


# Component 2 - Tabs to display FTE & wage amounts for each land use
def generate_tab(
    chartdata,
    land_uses_and_areas,
):

    seen_lus = []
    total_ftes = 0
    for trace in chartdata:
        if trace["name"] not in seen_lus:
            seen_lus.append(trace["name"])
            total_ftes += sum(trace["y"]) / len(trace["y"])
    total_area = sum([lu[1] for lu in land_uses_and_areas])
    fte_title = "Total annual FTEs"
    wages_title = "Total land area"

    def human_format(num, digits):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return "%.1f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])

    tab = html.Div(
        children=[
            html.Div(
                [
                    html.H4(wages_title),
                    html.H1(f"{human_format(total_area,1)} ha"),
                    html.H5(""),
                ],
                # className="six columns",
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "flex-grow": "1",
                    "background-color": "white",
                    # "border": "4px solid #41a0b0",
                    # "border-right": "2px solid #41a0b0",
                    # "border-radius": "4px",
                    "margin": 0,
                    "vertical-align": "middle",
                    "horizontal-align": "middle",
                },
            ),
            html.Div(
                [
                    html.H4(
                        fte_title,
                        style={
                            "vertical-align": "middle",
                            "horizontal-align": "middle",
                        },
                        id="fte-title-text",
                    ),
                    html.H1(
                        f"{total_ftes:.0f}",
                        style={
                            "vertical-align": "middle",
                            "horizontal-align": "middle",
                        },
                        id='fte-title-val'
                    ),
                    html.H5(
                        f"({0.9*total_ftes:.0f} - {1.1*total_ftes:.0f})",
                        # style={
                        #     "vertical-align": "bottom",
                        #     "horizontal-align": "middle",
                        # },
                        id="fte-confidence-interval",
                    ),
                ],
                # className="six columns",
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "flex-grow": "1",
                    "background-color": "white",
                    # "border": "4px solid #41a0b0",
                    # "border-left": "2px solid #41a0b0",
                    # "border-radius": "4px",
                    "margin": 0,
                    "vertical-align": "middle",
                    "horizontal-align": "middle",
                },
            ),
        ],
        id="wage-fte-div",
        style={
            "display": "flex",
            "flex-direction": "row",
            "justify-content": "space-between",
            "gap": "2rem",
        },
    )
    # tab = html.Div(
    #             children=[ftes_x_year,wages_x_year],
    #             style={
    #                 "width": "100%",
    #                 "height": "350px",
    #                 "margin-top": 0,
    #                 "padding": 0,
    #                 # "margin-top": "10px",
    #                 # "padding": "0px",
    #                 "position": "relative",
    #                 # "top": "0px",
    #             },
    #     id="wage-fte-div",

    return tab


def round_to_n_sf(x, n):
    return round(x, -int(floor(log10(x))) + (n - 1))


def sd_hours(lu_has, df, lus):
    df = df.copy()
    for k, lu in enumerate(lus):
        df.loc[df["land_use"] == lu, "hours"] = (
            df.loc[df["land_use"] == lu, "hours_ha"] * lu_has[k]
        )
    df = df.groupby("month_of_year").sum().reset_index()
    optimise = sum([((h - np.mean(df.hours)) ** 2) for h in df.hours]) / sum(df.hours)
    return optimise


def optimise_hours(df, selected_lus, minvals):
    # minimise the standard deviation of total hours by month by changing land area ha
    lus = list(pd.unique(df["land_use"]))
    lu_has = [5 for l in lus]
    lbs = [minvals[l] if l in selected_lus else 0 for l in lus]
    lu_has = optimize.minimize(
        sd_hours,
        lu_has,
        args=(df, lus),
        bounds=optimize.Bounds(lb=lbs, ub=100),  # array, keep original values min 5% lh
    )
    return lus, lu_has.x


def filterer(x, lus):
    row = [x[f"land_use_{i}"] for i in range(1, 5)]
    return all([lu in row for lu in lus])


def suggest_complementary(lus, lu_has, df, uses_selected, exclusions, stages, roles):
    complement["filterer"] = complement.apply(lambda x: filterer(x, lus), axis=1)
    filtered = complement[complement["filterer"]]

    filtered = filtered.loc[
        (~filtered["land_use_1"].isin(exclusions))
        & (~filtered["land_use_2"].isin(exclusions))
        & (~filtered["land_use_3"].isin(exclusions))
        & (~filtered["land_use_4"].isin(exclusions))
    ]

    if filtered.empty and len(lus) == 1:
        return "No matches found"
    elif filtered.empty:
        new_land_uses = lus
    else:
        row = filtered[filtered.score_to_min == min(filtered.score_to_min)]
        new_land_uses = [
            row[f"land_use_{i}"].iloc[0]
            for i in [1, 2, 3, 4]
            if row[f"land_use_{i}"].iloc[0]
            if type(row[f"land_use_{i}"].iloc[0]) == str
        ]
        # new_land_uses = [lu for lu in new_land_uses]

    # get new hours
    new_df = df[
        (df["land_use"].isin(new_land_uses))
        & (df["stage"].isin(stages))
        & (df["role"].isin(roles))
    ]
    total_land_area = sum(lu_has)
    # set lower bound at 5% of possible labour hours
    thresholder_df = new_df.groupby("land_use").mean(numeric_only=True).reset_index()
    failures = [n for n in new_land_uses if n not in thresholder_df["land_use"].tolist()]
    minvals = {}
    for lu in lus:
        adjust = thresholder_df.loc[thresholder_df["land_use"] == lu, "hours_ha"].iloc[
            0
        ] / sum(thresholder_df["hours_ha"])
        minvals[lu] = 20 * (1 - adjust)

    if failures != []:
        print(
            "Warning: land use mismatch between seasonal complementarity and worker database for: "
            + str(failures)
        )

    # optimise standard deviation of hours by month
    new_lus, lu_has = optimise_hours(new_df, lus, minvals)
    lu_has *= total_land_area / sum(lu_has)
    #     if new_lu==lu:
    #         rescale = lu_ha/new_lu_ha
    # lu_has = lu_has * rescale
    # collapse hours_ha by month_of_year
    new_df = (
        new_df.groupby(["month_of_year", "land_use"])["hours_ha"].mean().reset_index()
    )
    new_df = new_df.groupby("land_use")["hours_ha"].sum().reset_index()
    # new_df = new_df[new_df["land_use"] != lu]

    new_lus = list(new_lus)
    lu_has = list(lu_has)
    for new_lu, new_lu_ha in zip(new_lus, lu_has):
        if (new_lu_ha < 0.01 * sum(lu_has)) and (
            new_lu not in lus
        ):  # threshold on labour
            new_lus.remove(new_lu)
            lu_has.remove(new_lu_ha)
    new_ha = [round_to_n_sf(tup[0], 2) for tup in zip(lu_has, new_lus)]
    if new_lus == []:
        return "No matches found"
    # new_lus.remove(lu)
    # out =
    return new_lus, new_ha


# add graph layout
hours_x_month.update_layout(
    title={"text": "Seasonality in worker requirements", "x": 0.5},
    xaxis_title="Month",
    yaxis_title="Hours per hectare per month",
    font=dict(family="Arial", size=18, color="black"),
)
hours_x_month.update_yaxes(rangemode="tozero")

hours_x_month.update_layout(
    title={"text": "Seasonality in worker requirements", "x": 0.5}
)

# Interactive components
# Component 1 - Checkboxes for stages
stages_checklist = html.Div(
    children=[
        dcc.Checklist(
            options=[
                {
                    "label": html.Span(
                        "Start",
                        id="start-option",
                    ),
                    "value": "start",
                },
                {
                    "label": html.Span("Growing", id="growing-option"),
                    "value": "growing",
                },
                {
                    "label": html.Span("Established", id="established-option"),
                    "value": "established",
                },
            ],
            value=selected_stages,
            id="stages_checklist",
        ),
    ],
    style={"padding-bottom": "25px"},
    # id="stages_checklist",
)

start_tooltip_text = (
    "The initial planting or conversion year (Most land uses do not yet have data)"
)
growing_tooltip_text = "The stage between a crop's planting and full production"

graph_checklist = dcc.RadioItems(
    options=[
        {
            "label": html.Span("Seasonal (monthly data)", id="seasonal_option"),
            "value": "seasonal",
        },
        {
            "label": html.Span("30 year period (annual data)", id="longterm_option"),
            "value": "longterm",
        },
    ],
    value="seasonal",
    id="graph_checklist",
    style={"display": "inline-block"},
)

# Checkboxes for roles
roles_checklist = html.Div(
    children=[
        dcc.Checklist(
            options=[
                {
                    "label": html.Span(role.title(), id=f"{role}-option"),
                    "value": role,
                }
                for role in roles
            ],
            value=selected_roles,
            id="roles_checklist",
        )
    ],
    style={"padding-bottom": "25px"},
    className="hidden",
)

# Component 2 - Text input for land area to output data for in ha and button to submit input
land_area_input_text_box = dcc.Input(id="land_area", value=land_area_ha, type="number")
land_area_button = html.Button(
    id="submit-button", type="submit", children="Submit", className="btn"
)
suggestion_tooltip = dbc.Tooltip(
    "Suggest additional land uses, or rebalance hectares among existing land uses, to smooth out seasonal peaks",
    target="suggest-button",
    placement="right",
)
suggest_button = html.Button(
    id="suggest-button",
    type="submit",
    children=["Suggest", suggestion_tooltip],
    className="btn",
    style={"float": "right", "width": "30%", "margin-bottom": "10px", "padding": "0px"},
)
exclude_button = html.Button(
    id="exclude-button",
    type="submit",
    children="Regional restrictions",
    className="btn",
    style={"float": "right", "width": "30%", "margin-bottom": "10px", "padding": "0px"},
)

land_area_output_div = html.Div(id="output_div")

land_use_dropdown_div = dcc.Dropdown(
    id="land_use_dropdown",
    options=sorted(land_uses),
    value=selected_land_uses,
    style={
        "width": "100%",
        "display": "block",
        "height": "8vh",
        # "overflow-y": "scroll",
        # "overflow-x": "hidden",
        "vertical-align": "middle",
        "display": "inline-block",
        "padding-bottom": "25px",
    },
    multi=True,
)

land_use_exclusion_div = dcc.Dropdown(
    id="land_use_exclusion",
    options=sorted(land_uses),
    value=None,
    style={
        "width": "100%",
        "display": "block",
        "height": "8vh",
        # "overflow-y": "scroll",
        # "overflow-x": "hidden",
        "vertical-align": "middle",
        "display": "inline-block",
        "padding-bottom": "25px",
    },
    multi=True,
)


def layout():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            filter_pane_header_component,
                            html.Div(
                                [
                                    html.H5(
                                        "Land uses to include",
                                        style={
                                            "text-align": "left",
                                            "display": "inline",
                                        },
                                    ),
                                    suggest_button,
                                ],
                            ),
                            html.P(
                                "No suggestions found for this land use",
                                className="hidden",
                                id="no-match-text",
                            ),
                            land_use_dropdown_div,
                            html.Div(
                                [
                                    html.H5(
                                        "Land uses to exclude",
                                        style={
                                            "text-align": "left",
                                            "display": "inline",
                                        },
                                        id="exclude-title",
                                    ),
                                    exclude_button,
                                    dbc.Tooltip(
                                        "Restrict land uses from being suggested based on crop suitability data for your region",
                                        target="exclude-button",
                                        placement="top",
                                    ),
                                ]
                            ),
                            land_use_exclusion_div,
                            dbc.Tooltip(
                                "Select land uses to exclude from the suggested land use combinations",
                                target="exclude-title",
                                placement="top",
                            ),
                            html.H5(
                                "Stages to include",
                                style={"text-align": "left", "width": "350px"},
                            ),
                            stages_checklist,
                            dbc.Tooltip(
                                html.Span(
                                    "Crops have reached full production",
                                    id="established_text",
                                ),
                                target="established-option",
                                placement="right",
                            ),
                            dbc.Tooltip(
                                html.Span(
                                    "The initial planting or conversion year (Most land uses do not yet have data)",
                                    id="start_text",
                                ),
                                target="start-option",
                                placement="right",
                            ),
                            dbc.Tooltip(
                                html.Span(
                                    "The stage between a crop's planting and full production",
                                    id="growing_text",
                                ),
                                target="growing-option",
                                placement="right",
                            ),
                            dbc.Tooltip(
                                html.Span(
                                    "A Full Time Equivalent (FTE) is defined as 1840 hours per year, or 40 hours per week for 46 weeks.",
                                    id="fte_def",
                                ),
                                target="fte-title-text",
                                placement="top",
                            ),
                            dbc.Tooltip(
                                html.Span(
                                    "A Full Time Equivalent (FTE) is defined as 1840 hours per year, or 40 hours per week for 46 weeks.",
                                    id="fte_def2",
                                ),
                                target="fte-title-val",
                                placement="top",
                            ),
                            
                            html.P(
                                "Please select at least one role",
                                id="role-error",
                                className="hidden",
                            ),
                            # html.H5(
                            #     "Roles to include",
                            #     style={"text-align": "left"},
                            # ),
                            roles_checklist,
                            html.H5(
                                "Land use parameters",
                                style={"text-align": "left"},
                            ),
                            dcc.Loading(
                                html.Div(
                                    id="output-container",
                                    children=[
                                        html.Div(
                                            [
                                                (
                                                    "{} land area (ha)".format(lu)
                                                    if len("{}".format(lu)) <= 14
                                                    else "{}\nland area (ha)".format(lu)
                                                ),
                                                dcc.Input(
                                                    id="hectares-{}".format(lu),
                                                    type="number",
                                                    min=0,
                                                    value=10,
                                                    style={
                                                        "align": "right"
                                                    },  # "width": "350px"},
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                dcc.Dropdown(
                                                    id="start-{}".format(lu),
                                                    options=[{None: None}],
                                                    value=None,
                                                    multi=False,
                                                    clearable=False,
                                                    className="hidden",
                                                ),
                                                dcc.Dropdown(
                                                    id="end-{}".format(lu),
                                                    options=[{None: None}],
                                                    value=None,
                                                    multi=False,
                                                    clearable=False,
                                                    className="hidden",
                                                ),
                                            ],
                                        )
                                        for lu in land_uses
                                    ],
                                    style={
                                        "padding-bottom": "25px",
                                        "width": "100% !important",
                                        "height": "auto",
                                    },
                                ),
                            ),
                            # html.H5(
                            #     "Hourly wage rates",
                            #     style={"text-align": "left"},
                            # ),
                            # wage_rates,
                            # dbc.Tooltip(
                            #     "Carries out practical work on the land",
                            #     target="worker-option",
                            # ),
                            # dbc.Tooltip(
                            #     "Carries out management tasks related to primary production",
                            #     target="manager-option",
                            # ),
                            # dbc.Tooltip(
                            #     "Worker wage rate to use when calculating total wages",
                            #     target="worker-wage-title",
                            # ),
                            # dbc.Tooltip(
                            #     "Manager wage rate to use when calculating total wages",
                            #     target="manager-wage-title",
                            # ),
                            land_area_button,
                        ],
                        className="three columns",
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    dbc.Modal(
                                        html.Div(
                                            [
                                                html.H5("Describe your issue:"),
                                                dcc.Textarea(
                                                    id="feedback-input",
                                                    style={"width": "100%"},
                                                    required=True,
                                                ),
                                                html.H5(
                                                    "Attach a screenshot of the problem (optional):"
                                                ),
                                                dcc.Upload(
                                                    id="upload-image",
                                                    children=html.Div(
                                                        [
                                                            "Drag and Drop or ",
                                                            html.A("Select Image File"),
                                                        ]
                                                    ),
                                                    style={
                                                        "width": "100%",
                                                        "height": "250px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "margin-bottom": "20px",
                                                    },
                                                    # Allow multiple files to be uploaded
                                                    multiple=True,
                                                    className="",
                                                ),
                                                html.Div(id="output-image-upload"),
                                                dbc.Button(
                                                    "Submit feedback",
                                                    id="feedback-submit",
                                                ),
                                            ]
                                        ),
                                        size="xl",
                                        id="feedback-modal",
                                    ),
                                    dbc.Modal(
                                        [
                                            html.H5(
                                                "Restrict land use suggestions to crops suitable for your region:"
                                            ),
                                            dcc.Dropdown(
                                                options=[
                                                    {"label": r, "value": r}
                                                    for r in regional_exclusions[
                                                        "region"
                                                    ].unique()
                                                ],
                                                value="Auckland",
                                                multi=False,
                                                id="region-dropdown",
                                                clearable=True,
                                            ),
                                            html.H5("Restrict land uses to those:"),
                                            dcc.RadioItems(
                                                options=[
                                                    {
                                                        "label": "Already grown in selected region",
                                                        "value": "grown",
                                                    },
                                                    {
                                                        "label": "Already grown in selected region, or suitable for the region based on modelling",
                                                        "value": "suitable",
                                                    },
                                                    {
                                                        "label": "Either suitable for the selected region, or no data is available",
                                                        "value": "not_unsuitable",
                                                    },
                                                ],
                                                value="grown",
                                                id="suitability-radio",
                                            ),
                                            html.H5(
                                                "This will include the following land uses, everything else will be added to the exclusions list:"
                                            ),
                                            html.Div([], id="suitability-output"),
                                            dbc.Button(
                                                "Submit", id="suitability-submit"
                                            ),
                                        ],
                                        size="xl",
                                        id="exclusions-modal",
                                    ),
                                    dbc.Modal(
                                        [
                                            dbc.Button(
                                                "Download chart data",
                                                id="export-submit",
                                            ),
                                            dbc.Button(
                                                "Download full database",
                                                id="export-submit-full",
                                            ),
                                        ],
                                        size="m",
                                        id="export-modal",
                                    ),
                                    html.Div(
                                        children=[
                                            generate_tab(
                                                [],
                                                [],
                                            )
                                        ],
                                        id="tab_div",
                                        # style={"min-height": "200px", "padding": 0},
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Graph(
                                                figure=hours_x_month,
                                                id="seasonal_graph",
                                                config={"displayModeBar": False},
                                            ),
                                            # html.H5(
                                            #     "Graph type",
                                            #     style={
                                            #         "text-align": "left",
                                            #         "width": "350px",
                                            #     },
                                            # ),
                                            html.H5("Graph options"),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            graph_checklist,
                                                            dbc.Tooltip(
                                                                "Displays the monthly average requirements to show seasonality",
                                                                target="seasonal_option",
                                                                placement="left",
                                                            ),
                                                            dbc.Tooltip(
                                                                "Displays requirements for 30 years from land  use conversion with annual data",
                                                                target="longterm_option",
                                                                placement="left",
                                                            ),
                                                            dcc.RadioItems(
                                                                [
                                                                    "Total across land uses",
                                                                    "Separated by land use",
                                                                ],
                                                                "Separated by land use",
                                                                id="graph-type",
                                                                style={
                                                                    "display": "inline-block"
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dbc.Button(
                                                                        "Export data",
                                                                        id="export-button",
                                                                    ),
                                                                    dcc.Download(
                                                                        id="download"
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dbc.Button(
                                                                        "Report an issue",
                                                                        id="feedback-button",
                                                                    )
                                                                ],
                                                                id="feedback-div",
                                                            ),
                                                        ],
                                                        id="graph-type-div",
                                                    ),
                                                    html.Div(
                                                        html.P(
                                                            [
                                                                "This work is licensed under ",
                                                                html.A(
                                                                    "CC BY-SA 4.0",
                                                                    href="https://creativecommons.org/licenses/by-sa/4.0/",
                                                                    target="_blank",
                                                                    rel="license noopener noreferrer",
                                                                ),
                                                                ".",
                                                                html.Img(
                                                                    src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1",
                                                                    style={
                                                                        "height": "22px",
                                                                        "margin-left": "3px",
                                                                        "vertical-align": "text-bottom",
                                                                    },
                                                                ),
                                                                html.Img(
                                                                    src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1",
                                                                    style={
                                                                        "height": "22px",
                                                                        "margin-left": "3px",
                                                                        "vertical-align": "text-bottom",
                                                                    },
                                                                ),
                                                                html.Img(
                                                                    src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1",
                                                                    style={
                                                                        "height": "22px",
                                                                        "margin-left": "3px",
                                                                        "vertical-align": "text-bottom",
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        id="license-div",
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "justify-content": "space-between",
                                                    "flex-wrap": "wrap",
                                                },
                                            ),
                                        ],
                                        id="graph-div",
                                    ),
                                ],
                                className="eight columns",
                            ),
                        ],
                        style={"margin": "auto"},
                    ),
                ],
                className="row",
            ),
        ]
    )




@callback(Output("role-error", "className"), Input("roles_checklist", "value"))
def role_error(value):
    if not value:
        return "red-text"
    else:
        return "hidden"


def create_planting_table_row(
    lu, area, full_month_dd, value, existing_start=None, existing_end=None
):
    show = False
    valid_months = pd.unique(
        df.loc[df["land_use"] == lu, "planting_month"].dropna()
    ).tolist()
    planting_start_default = int(
        min(
            df.loc[
                (df["land_use"] == lu) & (df["default_planting_month"] == 1),
                "planting_month",
            ].dropna()
        )
    )
    planting_end_default = int(
        max(
            df.loc[
                (df["land_use"] == lu) & (df["default_planting_month"] == 1),
                "planting_month",
            ].dropna()
        )
    )
    plant_start_dd = dcc.Dropdown(
        id="start-{}".format(lu),
        options=[item for item in full_month_dd if item["value"] in valid_months],
        value=planting_start_default if not existing_start else existing_start,
        multi=False,
        clearable=False,
        className="planting-month-dd",
    )
    plant_end_dd = dcc.Dropdown(
        id="end-{}".format(lu),
        options=[item for item in full_month_dd if item["value"] in valid_months],
        value=planting_end_default if not existing_end else existing_end,
        multi=False,
        clearable=False,
        className="planting-month-dd",
    )
    input = (
        dcc.Input(
            id="hectares-{}".format(lu),
            type="number",
            min=0,
            value=area,
            persistence=True,
            persistence_type="session",
        ),
    )
    row = html.Tr(
        children=[
            html.Td(lu),
            html.Td(plant_start_dd),
            html.Td(plant_end_dd),
            html.Td(input),
        ]
    )
    if lu not in value:
        row.className = "hidden"
    else:
        show = True
    return row, show


def create_harvest_table_row(
    lu, area, full_month_dd, value, existing_start=None, existing_end=None
):
    show = False

    valid_months = pd.unique(
        df.loc[df["land_use"] == lu, "harvest_month"].dropna()
    ).tolist()
    harvest_months = df.loc[
        (df["land_use"] == lu) & (df["default_harvest_month"] == 1), "harvest_month"
    ].dropna()

    if harvest_months.empty:
        harvest_start_default = 1
        harvest_end_default = 3
    else:
        harvest_start_default = int(min(harvest_months))
        harvest_end_default = int(max(harvest_months))
    harvest_start_dd = dcc.Dropdown(
        id="start-{}".format(lu),
        options=[item for item in full_month_dd if item["value"] in valid_months],
        value=harvest_start_default if not existing_start else existing_start,
        multi=False,
        clearable=False,
        className="planning-month-dd",
    )
    harvest_end_dd = dcc.Dropdown(
        id="end-{}".format(lu),
        options=[item for item in full_month_dd if item["value"] in valid_months],
        value=harvest_end_default if not existing_end else existing_end,
        multi=False,
        clearable=False,
        className="planning-month-dd",
    )
    input = (
        dcc.Input(
            id="hectares-{}".format(lu),
            type="number",
            min=0,
            value=area,
            persistence=True,
            persistence_type="session",
        ),
    )
    row = html.Tr(
        children=[
            html.Td(lu),
            html.Td(harvest_start_dd),
            html.Td(harvest_end_dd),
            html.Td(input),
        ]
    )
    if lu not in value:
        row.className = "hidden"
    else:
        show = True
    return row, show


def create_neither_table_row(lu, area, months, value):
    show = False
    input = (
        dcc.Input(
            id="hectares-{}".format(lu),
            type="number",
            min=0,
            value=area,
            persistence=True,
            persistence_type="session",
        ),
    )
    hidden_start_dd = dcc.Dropdown(
        id="start-{}".format(lu),
        options=[{1: 1}],
        value=1,
        multi=False,
        clearable=False,
        className="hidden",
    )
    hidden_end_dd = dcc.Dropdown(
        id="end-{}".format(lu),
        options=[{1: 1}],
        value=1,
        multi=False,
        clearable=False,
        className="hidden",
    )
    row = html.Tr(
        children=[
            html.Td(lu),
            html.Td(["N/A", hidden_start_dd]),
            html.Td(["N/A", hidden_end_dd]),
            html.Td(input),
        ]
    )
    if lu not in value:
        row.className = "hidden"
    else:
        show = True
    return row, show


@callback(
    dash.dependencies.Output("output-container", "children"),
    Output("submit-button", "disabled"),
    Output("submit-button", "className"),
    [dash.dependencies.Input("land_use_dropdown", "value")],
    # hectares states
    [
        (
            State("start-{}".format(lu), "value"),
            State("end-{}".format(lu), "value"),
            State("hectares-{}".format(lu), "value"),
        )
        for lu in land_uses
    ],
)
# Use hectares as input and only initialise if None
def update_output(value, *land_use_states, land_uses=land_uses):

    output_list = []
    plant_table = html.Table(
        children=[
            html.Tr(
                children=[
                    html.Td("Land use"),
                    html.Td("Start of planting"),
                    html.Td("End of planting"),
                    html.Td("Area (ha)"),
                ],
                className="bold-text",
            )
        ],
        className="hidden",
    )
    harvest_table = html.Table(
        children=[
            html.Tr(
                children=[
                    html.Td("Land use"),
                    html.Td("Start of harvest"),
                    html.Td("End of harvest"),
                    html.Td("Area (ha)"),
                ],
                className="bold-text",
            )
        ],
        className="hidden",
    )
    full_month_dd = [
        {"label": "Jul", "value": 7},
        {"label": "Aug", "value": 8},
        {"label": "Sep", "value": 9},
        {"label": "Oct", "value": 10},
        {"label": "Nov", "value": 11},
        {"label": "Dec", "value": 12},
        {"label": "Jan", "value": 1},
        {"label": "Feb", "value": 2},
        {"label": "Mar", "value": 3},
        {"label": "Apr", "value": 4},
        {"label": "May", "value": 5},
        {"label": "Jun", "value": 6},
    ]

    for lu, (start, end, area) in zip(land_uses, land_use_states):
        if not df.loc[df["land_use"] == lu, "planting_month"].isna().all():
            plant_table.children.append(
                create_planting_table_row(
                    lu,
                    area,
                    full_month_dd,
                    value,
                    existing_start=start,
                    existing_end=end,
                )[0]
            )
            if create_planting_table_row(lu, area, full_month_dd, value)[1]:
                plant_table.className = "planting-table"

        elif not df.loc[df["land_use"] == lu, "harvest_month"].isna().all():
            harvest_table.children.append(
                create_harvest_table_row(
                    lu,
                    area,
                    full_month_dd,
                    value,
                    existing_start=start,
                    existing_end=end,
                )[0]
            )
            if create_harvest_table_row(lu, area, full_month_dd, value)[1]:
                harvest_table.className = "harvest-table"

        else:
            harvest_table.children.append(
                create_neither_table_row(lu, area, full_month_dd, value)[0]
            )
            if create_neither_table_row(lu, area, full_month_dd, value)[1]:
                harvest_table.className = "harvest-table"
    if not value:
        out = html.P(
            "Please select one or more land uses from the dropdown menu.",
            style={"font-weight": "bold", "color": "red"},
        )
        plant_table.className, harvest_table.className = "hidden", "hidden"
        output_div = html.Div(children=[out, plant_table, harvest_table])
        enable, button_class = True, "btn-disabled"

    else:
        enable, button_class = False, "btn"
        output_div = html.Div(children=[plant_table, harvest_table])
    return output_div, enable, button_class


@callback(
    Output("suggest-button", "className"),
    Output("suggest-button", "disabled"),
    Input("land_use_dropdown", "value"),
)
def suggest_button_disabler(uses):
    if len(uses) == 0:
        return "btn-disabled", True
    return "btn", False


# @app.callback(
#     Output("land_use_dropdown", "value"),
#     Output("land_use_exclusion", "value"),
#     Input("land_use_dropdown", "value"),
#     Input("land_use_exclusion", "value"),
# )
# def excluder(uses, exclusions):
#     if uses is None:
#         uses = []
#     if exclusions is None:
#         exclusions = []
#     for exclusion in exclusions:
#         if exclusion in uses:
#             uses.remove(exclusion)
#     return uses, exclusions


def get_exclusions(region, suitability, exclusions):
    if region is None:
        return exclusions
    if suitability == "grown":
        return regional_exclusions.loc[
            (regional_exclusions["region"] == region)
            & ~(regional_exclusions["total_suitability"] == "Suitable - already grown"),
            "land_use",
        ].tolist()
    elif suitability == "suitable":
        return regional_exclusions.loc[
            (regional_exclusions["region"] == region)
            & ~(
                (regional_exclusions["total_suitability"] == "Suitable - already grown")
                | (regional_exclusions["total_suitability"] == "Suitable - WWO")
            ),
            "land_use",
        ].tolist()
    elif suitability == "not_unsuitable":
        return regional_exclusions.loc[
            (regional_exclusions["region"] == region)
            & (regional_exclusions["total_suitability"] == "Unsuitable"),
            "land_use",
        ].tolist()


@callback(
    Output("suitability-output", "children"),
    Input("region-dropdown", "value"),
    Input("suitability-radio", "value"),
)
def display_inclusions(region, suitability):
    exclusions = get_exclusions(region, suitability, [])
    inclusions = sorted(
        [lu for lu in pd.unique(df["land_use"]) if lu not in exclusions]
    )
    return html.Div(
        children=[
            html.Ul(
                [html.Li(i) for i in inclusions[: int(len(inclusions) / 3)]],
                style={"width": "30%"},
            ),
            html.Ul(
                [
                    html.Li(i)
                    for i in inclusions[
                        int(len(inclusions) / 3) : int(2 * len(inclusions) / 3)
                    ]
                ],
                style={"width": "30%"},
            ),
            html.Ul(
                [html.Li(i) for i in inclusions[int(2 * len(inclusions) / 3) :]],
                style={"width": "30%"},
            ),
        ],
        style={"display": "flex"},
    )


@callback(
    Output("land_use_dropdown", "value"),
    Output("land_use_exclusion", "value"),
    [Output(f"hectares-{land_use}", "value") for land_use in land_uses],
    Output("no-match-text", "className"),
    Input("suggest-button", "n_clicks"),
    Input("land_use_dropdown", "value"),
    Input("land_use_exclusion", "value"),
    State("region-dropdown", "value"),
    State("suitability-radio", "value"),
    Input("suitability-submit", "n_clicks"),
    [State("stages_checklist", "value")],
    [State("roles_checklist", "value")],
    # [State("suggest-button", "disabled")],
    [
        (
            State(f"start-{land_use}", "value"),
            State(f"end-{land_use}", "value"),
            State(f"hectares-{land_use}", "value"),
        )
        for land_use in land_uses
    ],
    # don't run first go
    prevent_initial_call=True,
)
def suggester(
    n,
    uses_selected,
    exclusions,
    regions,
    suitability,
    suitability_submit,
    stages,
    roles,
    # submit_disabled,
    *land_use_states,
    land_uses=land_uses,
    df=df,
):
    starts = [start for start, end, area in land_use_states]
    ends = [end for start, end, area in land_use_states]
    land_use_states = [area for start, end, area in land_use_states]
    if exclusions is None:
        exclusions = []
    if ctx.triggered[0]["prop_id"] == "land_use_exclusion.value":
        if uses_selected is None:
            uses_selected = []
        for exclusion in exclusions:
            if exclusion in uses_selected:
                uses_selected.remove(exclusion)
        return uses_selected, exclusions, *land_use_states, "hidden"

    elif ctx.triggered[0]["prop_id"] == "land_use_dropdown.value":
        if uses_selected is None:
            uses_selected = []
        for use in uses_selected:
            if use in exclusions:
                exclusions.remove(use)
        return uses_selected, exclusions, *land_use_states, "hidden"
    elif ctx.triggered[0]["prop_id"] == "suitability-submit.n_clicks":
        exclusions = get_exclusions(regions, suitability, exclusions)
        for exclusion in exclusions:
            if exclusion in uses_selected:
                uses_selected.remove(exclusion)
        return uses_selected, exclusions, *land_use_states, "hidden"

    # if len(uses_selected) != 1:
    #     return uses_selected, "hidden"
    else:
        if type(stages) == str:
            stages = [stages]
        # fix with input
        land_area_ha = [
            land_use_states[land_uses.tolist().index(use)] for use in uses_selected
        ]
        starts = [starts[land_uses.tolist().index(use)] for use in uses_selected]
        ends = [ends[land_uses.tolist().index(use)] for use in uses_selected]
        # filter df by land use, stage, role
        new_df = df.loc[
            df["role"].isin(roles) & df["stage"].isin(stages),
            :,
        ]
        for lu, start, end in zip(uses_selected, starts, ends):
            check = new_df.loc[new_df["land_use"] == lu]
            if not check.planting_month.isna().all():
                new_df.loc[new_df["land_use"] == lu, "default_planting_month"] = 0
                if end >= start:
                    new_df.loc[
                        (
                            (new_df["land_use"] == lu)
                            & (new_df["planting_month"] >= start)
                            & (new_df["planting_month"] <= end)
                        ),
                        "default_planting_month",
                    ] = 1
                else:
                    new_df.loc[
                        (new_df["land_use"] == lu)
                        & (
                            (new_df["planting_month"] >= start)
                            | (new_df["planting_month"] <= end)
                        ),
                        "default_planting_month",
                    ] = 1
            elif not check.harvest_month.isna().all():
                new_df.loc[new_df["land_use"] == lu, "default_harvest_month"] = 0
                if end >= start:
                    new_df.loc[
                        (
                            (new_df["land_use"] == lu)
                            & (new_df["harvest_month"] >= start)
                            & (new_df["harvest_month"] <= end)
                        ),
                        "default_harvest_month",
                    ] = 1
                else:
                    new_df.loc[
                        (new_df["land_use"] == lu)
                        & (
                            (new_df["harvest_month"] >= start)
                            | (new_df["harvest_month"] <= end)
                        ),
                        "default_harvest_month",
                    ] = 1

        # if not tmp.planting_month.isna().all():
        #     tmp = tmp.loc[(tmp["planting_month"]>=start_month) & (tmp["planting_month"]<=end_month)] if end_month >= start_month else tmp.loc[(tmp["planting_month"]>=start_month) | (tmp["planting_month"]<=end_month)

        new_df = new_df[(new_df["default_planting_month"] != 0)]
        new_df = new_df[(new_df["default_harvest_month"] != 0)]
        # collapse hours_ha by month_of_year
        new_df = new_df.groupby(["month_of_year","role","land_use","stage"]).mean(numeric_only=True).reset_index()

        if (
            suggest_complementary(
                uses_selected,
                land_area_ha,
                new_df,
                uses_selected,
                exclusions,
                stages,
                roles,
            )
            == "No matches found"
        ):
            return tuple(
                [uses_selected] + [exclusions] + list(land_use_states) + ["red-text"]
            )

        out = suggest_complementary(
            uses_selected,
            land_area_ha,
            new_df,
            uses_selected,
            exclusions,
            stages,
            roles,
        )
        new_land_uses = out[0]
        new_ha = out[1]
        # convert land_use_states to list
        new_land_use_states = list(land_use_states)

        for i, land_use in enumerate(land_uses):
            if land_use in new_land_uses:
                new_land_use_states[i] = new_ha[new_land_uses.index(land_use)]
        # append new land use to uses_selected
        uses_selected = new_land_uses
        return tuple([uses_selected] + [exclusions] + new_land_use_states + ["hidden"])


@callback(
    Output("stages_checklist", "options"),
    Output("stages_checklist", "value"),
    Output("start_text", "children"),
    Output("growing_text", "children"),
    [Input("land_use_dropdown", "value")],
    [Input("graph_checklist", "value")],
    [Input("stages_checklist", "value")],
)
def filter_stages(uses, gtype, current_stages):
    strikethrough = {"text-decoration": "line-through"}
    nonstrikethrough = {"padding-left": 0}

    if gtype == "longterm" or len(uses) == 1:
        options = [
            {
                "label": html.Span(
                    [stage.title()],
                    id=f"{stage}-option",
                ),
                "value": stage,
                "id": stage,
            }
            for stage in stages
        ]
        if ctx.triggered[0]["prop_id"].split(".")[0] == "graph_checklist":
            value = stages if gtype == "longterm" else current_stages[-1:]
        else:
            value = current_stages if gtype == "longterm" else current_stages[-1:]

        start_text = (
            "The initial planting or conversion year (Most land uses do not yet have data)",
        )
        growing_text = "The stage between a crop's planting and full production"

    else:
        options = [
            {
                "label": html.Span(
                    [stage.title()],
                    style=nonstrikethrough if stage == "established" else strikethrough,
                    id=f"{stage}-option",
                ),
                "value": stage,
                "disabled": stage != "established",
            }
            for stage in stages
        ]
        value = ["established"]
        start_text = ("Not available for multiple land uses",)
        growing_text = "Not available for multiple land uses"

    return options, value, start_text, growing_text


def yearly_hours(df, new_stages, roles, land_uses_and_areas, graph_type):
    traces = []

    sum_hours = [0 for i in range(30)]
    collapsed_to_year_lu = subset_to_selections_yearly(
        df, [lu[0] for lu in land_uses_and_areas], roles, new_stages
    )
    ylim = 0
    only_one = len(land_uses_and_areas) == 1
    sum_meta = [["FTEs - " + role, 0] for role in roles]
    for k, lu in enumerate(land_uses_and_areas):
        meta = []
        # if clicks is not None and clicks > 0:

        hours = [0 for i in range(30)]
        for r, role in enumerate(roles):
            tmp = collapsed_to_year_lu.loc[
                (collapsed_to_year_lu["land_use"] == lu[0])
                & (collapsed_to_year_lu["role"] == role)
            ].copy()
            tmp["hours_total"] = tmp["hours_ha"] * lu[1]
            if role == "manager" and (
                tmp["management_category"].iloc[0] == 3
                or tmp["management_category"].iloc[0] == 4
            ):
                meta.append(
                    ("FTEs - manager", ["No data available" for i in range(len(tmp))])
                )
            else:
                meta.append(("FTEs - " + role, list(tmp.hours_total)))
            sum_meta[r][1] = (
                tmp.hours_total
                if k == 0
                else [sum(d) for d in zip(sum_meta[r][1], tmp.hours_total)]
            )
            # tmp = create_hours_by_month_for_one_land_use(
            # collapsed_to_year_lu, lu[0], lu[1], role
            # )
            for i in range(0, 30, 1):
                try:
                    hours_to_add = tmp.loc[
                        tmp["year_since_start"] == i, "hours_total"
                    ].iloc[0]
                except:
                    hours_to_add = 0
                hours[
                    i
                ] += hours_to_add  # tmp.loc[tmp["year_since_start"]==i,"hours_total"].iloc[0]

        ylim = max(ylim, max(hours))
        sum_hours = [
            sum(x) for x in itertools.zip_longest((sum_hours), hours, fillvalue=0)
        ]
        if tmp["management_category"].iloc[0] == 2:
            meta = [("FTEs - managers and workers", hours)]
        traces = triplicate_traces(traces, tmp.year_since_start, hours, lu[0], k, meta)

    if graph_type == "Total across land uses":
        traces = triplicate_traces(
            [], tmp.year_since_start, sum_hours, "Total all land uses", 0, sum_meta
        )
        # traces = [
        #     go.Scatter(x=tmp.year_since_start, y=sum_hours, name="Total all land uses",mode="lines+markers",error_y={"value":10,"type":"percent","color":"rgb(42, 63, 95)"})
        # ]
        ylim = max(sum_hours)
    return traces, 1.1 * ylim


def space_trace(traces, year_type):
    lmax = max([len(trace["name"]) for trace in traces])
    ymax = 0
    dupes = []

    for trace in traces:
        trace.y = tuple(
            [t / yearly_hours_x_fte * (1 if year_type else 12) for t in trace.y]
        )
        # convert hour values to ftes in place, but leave strings the same
        for i, row in enumerate(trace.customdata):
            for j, val in enumerate(row):
                if type(val) == float:
                    trace.customdata[i][j] = round(
                        val / yearly_hours_x_fte * (1 if year_type else 12), 1
                    )

        ymax = max(ymax, max(trace.y))
        # trace.meta = [trace["name"], ("Month" if not type else "Year")]
        if len(trace["name"]) == lmax:
            trace["name"] = trace["name"] + "    "
        if trace["name"] not in dupes:
            htempstr = f"<b>{trace['name']}</b><br>"
            for i, m in enumerate(trace.meta):
                htempstr += (
                    "%{" + f"meta[{i}]" + "}: %{" + f"customdata[{i}]" + "}      <br>"
                )
            trace.hovertemplate = htempstr + "<extra></extra>"
            dupes.append(trace["name"])
        else:
            trace.hoverinfo = "skip"
        if not year_type:
            trace.x = np.concatenate((trace.x[6:], trace.x[:6]))
            trace.y = np.concatenate((trace.y[6:], trace.y[:6]))
            trace.meta = np.concatenate((trace.meta[6:], trace.meta[:6]))
            trace.customdata = np.concatenate(
                (trace.customdata[6:], trace.customdata[:6])
            )
    return traces, lmax, ymax * 1.15


def triplicate_traces(traces, series, hours, name, i, meta):
    hours_lower = [h * 0.9 for h in hours]
    hours_upper = [h * 1.1 for h in hours]
    customdata = list(zip(*[m[1] for m in meta]))
    meta = [m[0] for m in meta]
    traces.append(
        go.Scatter(
            x=series,
            y=hours,
            name=name,
            mode="markers+lines",
            line={"width": 2, "color": COLORS[i]},
            customdata=customdata,
            meta=meta,
        )
    )
    traces.append(
        go.Scatter(
            x=series,
            y=hours_lower,
            name=name,
            mode="lines",
            line={"width": 0, "color": COLORS[i]},
            fillcolor="rgba(0,100,80,0)",
            fill="tonexty",
            # hovertemplate="<br>" + "%{y}<br>",
            hoverinfo="skip",
            showlegend=False,
            customdata=customdata,
            meta=meta,
        )
    )
    traces.append(
        go.Scatter(
            x=series,
            y=hours_upper,
            name=name,
            mode="lines",
            line={"width": 0, "color": COLORS[i]},
            fillcolor=COLORS_TRANSPARENT[i],
            fill="tonexty",
            # hovertemplate="<br>" + "%{y}<br>",
            hoverinfo="skip",
            showlegend=False,
            customdata=customdata,
            meta=meta,
        )
    )
    return traces

@callback(
        Output("fte_def", "children"),
        Output("fte_def2", "children"),
        [Input("graph_checklist", "value")])
def update_tooltip(gtype):
    print(gtype)
    if gtype == "seasonal":
        ttiptext = "A Full Time Equivalent (FTE) is defined as 1840 hours per year, or 40 hours per week for 46 weeks. This value is the average across the one year for the selected stage(s)."
    else:
        ttiptext = "A Full Time Equivalent (FTE) is defined as 1840 hours per year, or 40 hours per week for 46 weeks. This value is the average across the 30 year period for the selected stage(s)."
    return ttiptext, ttiptext


# need one app callback per output
@callback(
    Output("tab_div", "children"),
    Output("seasonal_graph", "figure"),
    [Input("submit-button", "n_clicks")],
    [
        State("stages_checklist", "value")
    ],  # If this is Input, 'graph' type behaviour is good but it updates the graph without you hitting submit
    [Input("stages_checklist", "options")],
    [State("roles_checklist", "value")],
    # [State("worker-wage", "value")],
    # [State("manager-wage", "value")],
    [Input("graph-type", "value")],
    [Input("graph_checklist", "value")],
    [Input("land_use_dropdown", "value")],
    [
        (
            State(f"start-{land_use}", "value"),
            State(f"end-{land_use}", "value"),
            State(f"hectares-{land_use}", "value"),
        )
        for land_use in land_uses
    ],
    prevent_initial_call=False,
)
def update_graph(
    clicks,
    new_stages,
    _,
    roles,
    graph_type,
    graph_yearly,
    land_uses_input,
    *land_use_states,
    land_uses=land_uses,
):
    if type(new_stages) == str:
        new_stages = [new_stages]
    collapsed_to_month_lu = subset_to_selections_and_collapse(
        df, land_uses_input, roles, new_stages
    )
    yearly = graph_yearly == "longterm"
    ylim = 0
    traces = []
    land_uses_and_areas = []
    start_months = [s[0] for s in land_use_states]
    end_months = [s[1] for s in land_use_states]
    land_use_states = [s[2] for s in land_use_states]
    sum_hours = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    to_bar = pd.DataFrame()
    for i, lu in enumerate(land_uses_input):
        meta = []
        # if clicks is not None and clicks > 0:
        land_area_ha = land_use_states[land_uses.tolist().index(lu)]
        start_month = start_months[land_uses.tolist().index(lu)]
        end_month = end_months[land_uses.tolist().index(lu)]
        land_uses_and_areas.append((lu, land_area_ha))
        hours = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for role in roles:
            if start_month and end_month:
                tmp = create_hours_by_month_for_one_land_use(
                    collapsed_to_month_lu,
                    lu,
                    land_area_ha,
                    role,
                    start_month=start_month,
                    end_month=end_month,
                )
            else:
                tmp = create_hours_by_month_for_one_land_use(
                    collapsed_to_month_lu, lu, land_area_ha, role
                )
            hours = [
                sum(x)
                for x in itertools.zip_longest(
                    (hours), list(tmp.hours_total), fillvalue=0
                )
            ]
            if role == "manager" and (
                tmp["management_category"].iloc[0] == 3
                or tmp["management_category"].iloc[0] == 4
            ):
                meta.append(
                    ("FTEs - manager", ["No data available" for i in range(len(tmp))])
                )
            else:
                meta.append(("FTEs", list(tmp.hours_total)))

            to_bar = pd.concat([to_bar, tmp], axis=0)
        ylim = max(ylim, max(hours))
        sum_hours = [
            sum(x) for x in itertools.zip_longest((sum_hours), hours, fillvalue=0)
        ]
        if tmp["management_category"].iloc[0] == 2:
            meta = [("FTEs", hours)]
        meta.append(("Task", tmp["task"]))

        if len(set(hours)) <= 1:
            name = f"{lu} (no seasonal data) "
            tmp["land_use"] = tmp.apply(
                lambda x: x["land_use"] + " (no seasonal data)", axis=1
            )
        else:
            name = lu
        traces = triplicate_traces(traces, tmp.month_short_name, hours, name, i, meta)

    if yearly:
        traces, ymax = yearly_hours(
            df, new_stages, roles, land_uses_and_areas, graph_type
        )
        traces, lmax, ymax = space_trace(traces, yearly)
        graph_dict = {
            "data": traces,
            "layout": go.Layout(
                title={"text": "Long-term trends in worker requirements", "x": 0.5},
                height=610,
                xaxis_title="Year",
                yaxis_title="FTEs needed during year",
                yaxis_hoverformat=".0f",
                hovermode="x unified",
                hoverlabel_font_size=10,
                hoverlabel_align="left",
                yaxis_range=[0, ymax],
                font=dict(family="Arial", size=18, color="black"),
                showlegend=True,
            ),
        }

    else:
        if graph_type == "Total across land uses":
            combroleflag = [False for l in land_uses_input]
            sdf = to_bar.groupby("month_short_name").sum().reset_index()
            customdata = []
            for i, lu in enumerate(land_uses_input):
                subset = to_bar.loc[to_bar.land_use == lu]

                if (
                    subset["management_category"].iloc[0] == 3
                    or subset["management_category"].iloc[0] == 4
                ):
                    worker_df = subset.loc[(subset.role == "worker"), "hours_total"]
                    manager_df = worker_df.apply(lambda x: "No data available")
                    d = {"worker": worker_df, "manager": manager_df}
                    customdata.append([d[role] for role in roles])
                elif subset["management_category"].iloc[0] == 1:
                    customdata.append(
                        [
                            subset.loc[(subset.role == role), "hours_total"]
                            for role in roles
                        ]
                    )
                else:

                    combroleflag[i] = True

                    combval = (
                        subset.groupby(["month_short_name"], sort=False)
                        .sum()
                        .reset_index()
                    )
                    # print(combval)
                    customdata.append([combval["hours_total"]])
            to_bar = (
                to_bar.groupby(["month_short_name", "land_use"], sort=False)
                .sum()
                .reset_index()
            )

            sdf["error"] = sdf.apply(
                lambda x: x["hours_total"] / (10 * (yearly_hours_x_fte) / 12), axis=1
            )
            sdf.sort_values("month_of_year", inplace=True)
            to_bar["error"] = to_bar.apply(
                lambda x: dict(value=0, visible=False), axis=1
            )
            to_bar["error"][-12:] = list(sdf["error"])
            fig = px.bar(
                to_bar,
                x="month_short_name",
                y="hours_total",
                color="land_use",
                error_y="error",
            )
            for i in range(len(fig.data)):

                fig.data[i].meta = (
                    ["FTEs - " + r for r in roles]
                    if not combroleflag[i]
                    else ["FTEs - managers and workers"]
                )
                fig.data[i].customdata = list(zip(*customdata[i]))

            fig.update_layout(
                title={"text": "Seasonality in worker requirements", "x": 0.5},
                height=610,
                xaxis_title="Year",
                yaxis_title="FTEs needed during month",
                yaxis_hoverformat=".0f",
                legend_title=None,
                plot_bgcolor="white",
                # hovermode="x unified",
                # hoverlabel_namelength=-1,
                # hoverlabel_font_size=10,
                # hoverlabel_align="left",
                yaxis_range=[
                    0,
                    1.15 * max(sdf["hours_total"]) / (yearly_hours_x_fte / 12),
                ],
                font=dict(family="Arial", size=18, color="black"),
                showlegend=True,
            )
            fig.data = space_trace(fig.data, yearly)[0]
            tabs = generate_tab(fig.data, land_uses_and_areas)
            return tabs, fig
        traces, lmax, ylim = space_trace(traces, yearly)
        graph_dict = {
            "data": traces,
            "layout": go.Layout(
                title={"text": "Seasonality in worker requirements", "x": 0.5},
                height=610,
                xaxis_title="Month",
                yaxis_title="FTEs needed during month",
                yaxis_hoverformat=".0f",
                hovermode="x unified",
                # hoverlabel_namelength=-1,
                hoverlabel_font_size=10,
                hoverlabel_align="left",
                yaxis_range=[0, 1.1 * ylim],
                font=dict(family="Arial", size=18, color="black"),
                showlegend=True,
            ),
        }
    tabs = generate_tab(graph_dict["data"], land_uses_and_areas)

    return tabs, graph_dict


@callback(
    Output("feedback-modal", "is_open"),
    [
        Input("feedback-button", "n_clicks"),
    ],
    [State("feedback-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(
    n0,
    is_open,
):
    if n0:
        return not is_open

    return is_open


@callback(
    Output("exclusions-modal", "is_open"),
    [
        Input("exclude-button", "n_clicks"),
        Input("suitability-submit", "n_clicks"),
    ],
    [State("exclusions-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(
    n0,
    n1,
    is_open,
):
    if n0 or n1:
        return not is_open
    return is_open


@callback(
    Output("export-modal", "is_open"),
    Output("download", "data"),
    [
        Input("export-button", "n_clicks"),
        Input("export-submit", "n_clicks"),
        Input("export-submit-full", "n_clicks"),
        State("seasonal_graph", "figure"),
    ],
    [State("export-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(
    n0,
    n1,
    n2,
    chartdata,
    is_open,
):
    if ctx.triggered[0]["prop_id"].split(".")[0] == "export-button":
        return not is_open, None
    elif ctx.triggered[0]["prop_id"].split(".")[0] == "export-submit-full":
        return not is_open, dcc.send_data_frame(df.to_csv, "LandFTEsDatabase.csv")
    res = None
    seen_lus = []
    for trace in chartdata["data"]:
        xdata = trace["x"]
        ftes = trace["y"]
        lu = trace["name"]
        xname = "Month" if len(xdata) == 12 else "Year"
        if seen_lus == []:
            res = pd.DataFrame({xname: xdata, lu: ftes})
        elif lu in seen_lus:
            continue
        else:
            res[lu] = ftes
        seen_lus.append(lu)
        if len(res.columns) > 2:
            res["Total"] = res[seen_lus].sum(axis=1)
            if res.columns[-1] != "Total":
                new_column_order = [col for col in res.columns if col != "Total"] + [
                    "Total"
                ]
                res = res[new_column_order]

    return not is_open, dcc.send_data_frame(
        res.to_csv, "LandFTEsChartData.csv", index=False
    )


def parse_contents(contents, filename, date):
    return html.Div(
        [
            dbc.Button(
                "x", id="close-button", color="danger", className="btn-top-right"
            ),
            html.H6(filename),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents, style={"max-width": "250px", "max-height": "250px"}),
        ],
        style={
            "max-height": "350px",
            "width": "100%",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "position": "relative",
        },
    )


@callback(
    Output("output-image-upload", "children"),
    Output("upload-image", "className"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children, "hidden"
    return None, ""


@callback(
    Output("upload-image", "contents"),
    Input("close-button", "n_clicks"),
    State("upload-image", "contents"),
    prevent_initial_call=True,
)
def close_modal(n_clicks, contents):
    if n_clicks:
        return None
    return contents


@callback(Output("feedback-submit", "className"), Input("feedback-input", "value"))
def disable_submit(text):
    if (text is not None) or (text == ""):
        return ""
    return "btn-disabled"


def to_str(obj):
    if type(obj) == Str:
        return obj
    elif type(obj) == list:
        streturn = ""
        for o in obj:
            streturn += str(o) + ", "
        return streturn[:-2]
    else:
        return str(obj)


@callback(
    Output("feedback-button", "n_clicks"),
    Input("feedback-submit", "n_clicks"),
    State("feedback-button", "n_clicks"),
    State("land_use_dropdown", "value"),
    State("stages_checklist", "value"),
    State("roles_checklist", "value"),
    State("graph_checklist", "value"),
    State("graph-type", "value"),
    State("suggest-button", "n_clicks"),
    State("feedback-input", "value"),
    State("upload-image", "contents"),
    [State(f"hectares-{land_use}", "value") for land_use in land_uses],
    prevent_initial_call=True,
)
def report_state_and_feedback(
    n0,
    n1,
    land_uses_input,
    stages_input,
    roles_input,
    graph_input,
    graph_type,
    suggest_button_clicks,
    user_feedback,
    image,
    *land_use_states,
    land_uses=land_uses,
):
    filtered_land_use_states = [
        state
        for land_use, state in zip(land_uses, land_use_states)
        if land_use in land_uses_input
    ]
    dbstate = ""
    dbstate += "land uses: " + to_str(land_uses_input) + "\n"
    dbstate += "land use areas: " + to_str(filtered_land_use_states) + "\n"
    dbstate += "stages: " + to_str(stages_input) + "\n"
    dbstate += "roles: " + to_str(roles_input) + "\n"
    dbstate += "graph: " + to_str(graph_input) + "\n"
    dbstate += "graph type: " + to_str(graph_type) + "\n"
    dbstate += "suggest button clicks: " + to_str(suggest_button_clicks) + "\n"

    msg = MIMEMultipart()
    email_receiver = "luc@scarlatti.co.nz"
    msg["Subject"] = "LUC dashboard feedback"
    msg["From"] = email_params["email"]
    msg["To"] = email_receiver  # receiver email
    msg.attach(
        MIMEText(
            "User feedback:\n" + user_feedback + "\n \n Dashboard state:\n" + dbstate
        )
    )
    if image is not None:  # TO FIX
        # save image as appropriate file type
        image = image[0]
        image = image.split(",")[1]
        image = base64.b64decode(image)

        msg.attach(MIMEImage(image))
    server = smtplib.SMTP("smtp-mail.outlook.com", 587)
    server.starttls()  # encrypted connection
    server.login(email_params["email"], email_params["password"])
    server.sendmail(email_params["email"], email_receiver, msg.as_string())
    server.quit()
    return n1 + 1


# Run App
