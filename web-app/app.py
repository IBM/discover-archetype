import os
import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from flask import Flask, flash, request, redirect, Response
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = set(['zip'])

app = Flask(__name__, instance_relative_config=True)
app.config.from_pyfile('config.py', silent=True)


def initialize_dash_app():
    from pages.dashboard.layout import layout as archetype_layout
    from pages.dashboard.callbacks import register_callbacks as ac_reg
    from pages.home.layout import layout as home_layout
    from pages.upload.layout import layout as upload_layout
    from pages.upload.callbacks import register_callbacks as up_reg

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

    with app.app_context():
        dash_app.title = 'Archetype Discovery'
        ac_reg(dash_app)
        up_reg(dash_app)

    # Update the index
    @dash_app.callback(dash.dependencies.Output('page-content', 'children'),
                  [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/archetypes':
            return archetype_layout
        elif pathname == '/upload':
            return upload_layout
        else:
            return home_layout


initialize_dash_app()


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_corpus():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return {'error': 'POST request does not contain file.'}, 400
#         file = request.files['file']
#         if file.filename == '':
#             return {'error': 'Empty filename given'}, 400
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(path)
#             extract_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
#             util.unzip_corpus(path, extract_directory)
#             return {'msg': 'Corpus successfully uploaded'}, 200
#     return '''
#     <!doctype html>
#     <title>Upload Corpora</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''
