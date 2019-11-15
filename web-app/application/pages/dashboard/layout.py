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
                html.P('Perform archetypal analysis on a corpus:'),
                html.Ol([
                    html.Li([
                        html.Strong('Natural Language Understanding'),
                        html.Ul([
                            html.Li(
                                'Select a corpus that has been analyzed by '
                                'IBM Watson Natural Language Understanding.'
                            ),
                            html.Li(
                                'Select what output variables to discover '
                                'archetypes from (i.e. concepts, keywords, '
                                'and entities)'
                            ),
                        ]),
                    ]),
                    html.Li([
                        html.Strong('Archetypal Analysis'),
                        html.Ul([
                            html.Li(
                                'From the selected corpus data, archetypes '
                                'will be created by clustering data over '
                                'the selected variable using NMF '
                                '(Non-Negative Matrix Factorization).'
                            ),
                            html.Li(
                                'Variables are mapped onto the '
                                'archetypes/clusters.'
                            ),
                        ]),
                    ]),
                ]),
            ]),
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
                                {'label': 'Concepts', 'value': 'concepts'},
                                {'label': 'Keywords', 'value': 'keywords'},
                                {'label': 'Entities', 'value': 'entities'},
                            ],
                            value='concepts',
                        )
                    ]),
                    dbc.Col([
                        html.Label('#Archetypes',
                                   style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='NoA',
                            options=[{'label': k, 'value': k}
                                     for k in range(2, 16)],
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
