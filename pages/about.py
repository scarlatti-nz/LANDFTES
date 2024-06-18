import dash
from dash import dcc, html
from dash.dependencies import Output, State, Input
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/about", name="About", title="About")


def layout():
    return html.Div(
        children=[
            html.Div(
                [
                    html.H3("About the dashboard", style={"font-weight": "600"}),
                    html.P(
                        children=[
                            "This dashboard is designed to be a tool for land owners, stewards, managers, catchment groups, hapū, and rural professionals to estimate the number of full-time equivalents (FTEs) needed for different land-use scenarios, as well as how this workforce requirement varies seasonally. Additionally, the dashboard can suggest land uses which complement a selected land use(s) to smooth out seasonal variance in workforce requirements. If you happen to find a bug in the dashboard, please let us know via the “Report an issue” button, and we’ll do our best to get it sorted quickly. If you have any further questions about the dashboard, feel free to contact us at ",
                            html.A(
                                href="mailto:luc@scarlatti.co.nz",
                                children="luc@scarlatti.co.nz",
                            ),
                        ]
                    ),
                    html.Br(),
                    html.H3("About the project", style={"font-weight": "600"}),
                    html.P(
                        "This project was made possible by funding from the Our Land and Water National Science Challenge – in particular the project ‘Workforce Implications of Land-Use Change’.\
                                As such, this tool and its desired outcomes align with Our Land and Water’s vision: that in future New Zealand resembles a diverse ‘mosaic’ of land uses that are more resilient,\
                                lead to healthier land and water, and provide more income security than today. "
                    ),
                    html.Br(),
                    html.A(
                        "More information about Our Land and Water",
                        href="https://ourlandandwater.nz/about-us",
                    ),
                    html.Br(),
                    html.Br(),
                    html.A(
                        "More information about the project",
                        href="https://ourlandandwater.nz/project/workforce-implications-of-land-use-change",
                    ),
                    html.Br(),
                    html.Br(),
                    html.H3("Disclaimer", style={"font-weight": "600"}),
                    html.P(
                        "While every effort has been made to ensure the information in this dashboard is accurate, Scarlatti/Our Land and Water does not accept any responsibility or liability for error of fact, omission, interpretation or opinion that may be present, nor for the consequences of any decisions based on this information."
                    ),
                    html.Br(),
                    html.Br(),
                    html.H3("License", style={"font-weight": "600"}),
                    html.P(
                        [
                            "This work is licensed under ",
                            html.A(
                                "CC BY-SA 4.0",
                                href="https://creativecommons.org/licenses/by-sa/4.0/",
                                target="_blank",
                                rel="license noopener noreferrer"
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
                    html.Br(),
                    html.Br(),
                ],
                style={"margin": "auto", "width": "60%"},
            )
        ],
        style={"background-color": "#f9f9f9", "padding": "2rem"},
    )
