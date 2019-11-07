import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from application.models import Corpus


def layout():
    corpus_options = []
    corpora = Corpus.query.all()
    for corpus in corpora:
        corpus_options.append({
            'label': '{} (ID: {})'.format(corpus.name, corpus.id),
            'value': corpus.id
        })
    return html.Div(
        html.Div([
            html.Div([
                html.H1(
                    children='Archetype Clustering',
                    className="nine columns",
                    style={
                        'marginTop': 20,
                        'marginRight': 20
                    },
                ),

                dcc.Markdown(
                    children='''
                        Archetypal Analysis of Medical Dictations. Process:
                        1. **Natural Language Understanding**:
                            - Dictations are analyzed by IBM Watson Natural
                              Language Understanding.
                            - Output variables: keywords, entities, concepts
                              and categories.
                        2. **Archetypal Analysis**:
                            - Create Archetypes: Cluster data over variables,
                              using NMF Non-zero Matrix Factorization
                        ''',
                    className='nine columns')
            ], className="row"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label('Corpus', style={'fontWeight': 'bold'}),
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
                        html.Label('Variables', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='Var',
                            options=[
                                {'label': 'Keywords', 'value': 'keywords'},
                                {'label': 'Entities', 'value': 'entities'},
                                {'label': 'Concepts', 'value': 'concepts'},
                                {'label': 'Categories', 'value': 'categories'},
                            ],
                            value='keywords',
                        )
                    ]),
                    dbc.Col([
                        html.Label('#Archetypes',
                                   style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='NoA',
                            options=[{'label': k, 'value': k}
                                     for k in range(2, 100)],
                            value=6,
                            multi=False
                        )
                    ]),
                    dbc.Col([
                        html.Label('Cut at', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='Threshold',
                            options=[{'label': str(k)+'%', 'value': k/100}
                                     for k in range(1, 99)],
                            value=0.1,
                            multi=False
                        )
                    ])
                ])
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='variables-heatmap'
                    )
                ])
            ]),
         ])
    )
