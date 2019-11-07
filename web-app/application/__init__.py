import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask

from application.extensions import db
from application.pages.dashboard.layout import layout as archetype_layout
from application.pages.dashboard.callbacks import register_callbacks as ac_reg
from application.pages.home.layout import layout as home_layout
from application.pages.upload.layout import layout as upload_layout
from application.pages.upload.callbacks import register_callbacks as up_reg


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile('config.py', silent=True)
    register_extensions(app)

    external_stylesheets = [
        dbc.themes.BOOTSTRAP
    ]
    meta_viewport = {
        'name': 'viewport',
        'content': 'width=device-width, initial-scale=1, shrink-to-fit=no'
    }
    dash_app = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/',
                         external_stylesheets=external_stylesheets,
                         meta_tags=[meta_viewport])

    dash_app.config.suppress_callback_exceptions = True

    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div([
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Upload", href="/upload")),
                    dbc.NavItem(dbc.NavLink("Archetypes", href="/archetypes")),
                    dbc.NavItem(dbc.NavLink("Match", href="#"))
                ],
                brand="Archetype Discovery",
                brand_href="/",
                color="dark",
                dark=True,
            ),
            html.Div(id='page-content', className='container')
        ])
    ])

    # Update the index
    @dash_app.callback(dash.dependencies.Output('page-content', 'children'),
                       [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/archetypes':
            return archetype_layout()
        elif pathname == '/upload':
            return upload_layout()
        else:
            return home_layout()

    with app.app_context():
        dash_app.title = 'Archetype Discovery'
        ac_reg(dash_app)
        up_reg(dash_app)

        # Create tables for our models
        db.create_all()

        return app


def register_extensions(app):
    db.init_app(app)
