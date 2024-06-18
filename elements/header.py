from dash import html, dcc

olw_logo = "assets/olwlogofull.svg"
scarlatti_logo = "assets/White Scarlatti secondary logo no tagline.svg"


def layout():
    return html.Div(
        # className="three columns",
        children=[
            html.Div(
                [
                    html.Img(
                        id="olw_logo",
                        src=olw_logo,
                        style={
                            # "height": "4%",
                            # "width": "10%",
                            "horizontal-align": "right",
                            # "display": "inline-block",
                        },
                    ),
                    html.Div(
                        html.H1(
                            html.B(
                                "Worker requirements by land use"
                            ),
                            style={"text-align": "center"},
                        ),
                        id="header-title",
                    ),
                    tabs(),
                    html.Img(
                        id="scarlatti_logo",
                        src=scarlatti_logo,
                        style={
                            # #"height": "12%",
                            # "width": "12%",
                            "horizontal-align": "right",
                            # "display": "inline-block",
                        },
                    ),
                ],
                className="header-div",
            ),
        ],
        style={"horizontalAlign": "right"},
    )


def tabs():
    return html.Div(
        id="header-links",
        children=[
            html.Div(
                dcc.Link(
                    "Home",
                    href="/",
                    className="header-link",
                )
            ),
            html.Div(
                dcc.Link(
                    "About",
                    href="about",
                    className="header-link",
                )
            ),
        ],
    )
