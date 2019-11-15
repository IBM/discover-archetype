import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from application.models import Corpus


def layout():
    corpus_options = []
    corpora = Corpus.query.filter(Corpus.status == 'ready').all()
    for corpus in corpora:
        corpus_options.append({
            'label': '{} (ID: {})'.format(corpus.name, corpus.id),
            'value': corpus.id
        })
    return html.Div([
        html.H3('Match New Document'),
        html.P('Analyze a new document and map it onto the archetypes.'),
        dbc.Row([
            dbc.Col([
                html.Label('Corpus to Analyze Against',
                           style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='corpus-dropdown',
                    options=corpus_options,
                    value=(corpus_options[0]['value']
                           if len(corpus_options) else None)
                )
            ], width=4)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col([
                html.Label('Variables',
                           style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='type',
                    options=[
                        {'label': 'Concepts', 'value': 'concepts'},
                        {'label': 'Keywords', 'value': 'keywords'},
                        {'label': 'Entities', 'value': 'entities'},
                    ],
                    value='concepts',
                )
            ]),
            dbc.Col([
                html.Label('# of Archetypes',
                           style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='num-archetype',
                    options=[{'label': k, 'value': k}
                             for k in range(2, 16)],
                    value=6,
                    multi=False
                )
            ]),
        ], className='mb-4'),
        html.Label('Document Text', style={'fontWeight': 'bold'}),
        dbc.Textarea(className='mb-3',
                     id='document-text',
                     placeholder='Enter text here',
                     rows=10),
        dbc.Button('Submit',
                   color='primary',
                   id='submit-button'),
        html.Div([
            dcc.Graph(
                id='output-similarity-graph',
                style={'margin-right': 'auto',
                       'margin-left': 'auto',
                       'width': '50%'}
            )
        ]),
        html.Div([
            dcc.Graph(
                id='output-archetypes-graph'
            )
        ])
    ], className='mt-4 mb-4')
