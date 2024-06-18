import dash

style_sheets = "assets/layout.css"
app = dash.Dash(
    __name__,
    # server=server,
    external_stylesheets=[style_sheets],
    use_pages=True,
)


server = app.server
app.config["suppress_callback_exceptions"] = True

from ast import Str
import dash
from dash import dcc, html
from dash.dependencies import Output, State, Input
from flask import Flask
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
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
import pages.main as main
import pages.about as about
# from tabs import layout as tabs_layout
from elements.header import layout as header_layout

# from app import app
# Initiate App
# Style files

# app.layout = html.Div([dcc.Store(id="first_visit", storage_type="local", data=0),
#     dcc.Location(id="url", refresh=True)],id="page-content")
app.layout = html.Div(
    children=[
        header_layout(),
        dash.page_container,
    ]
)
# Layout

# @app.callback(
#     Output("url", "pathname"),
#     Output("first_visit", "data"),
#     State("first_visit", "data"),
#     Input("first_visit","storage_type")
# )
# def first_visit(data,_):
#     print(data)
#     return "/", 1 # remove when adding landing page
#     if data==0:
#         return "/", 1
#     else:
#         return "/dashboard", 1

# @app.callback(
#     Output("page-content", "children"),
#     Input("url", "pathname"),
# )
# def return_layout(url):
#     print(url)
#     if url=="/":
#         return main.layout
#     elif url=="/about":
#         return about.layout
#     else:
#         return html.H1('404 Page Not Found')

server = app.server
if __name__ == "__main__":
    server.run(host="127.0.0.1", debug=False, port=5050)
