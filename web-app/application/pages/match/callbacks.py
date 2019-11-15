from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analyzer import analyze_text


def register_callbacks(dash_app):

    @dash_app.callback(
        [Output('output-similarity-graph', 'figure'),
         Output('output-archetypes-graph', 'figure')],
        [Input('submit-button', 'n_clicks')],
        [State('corpus-dropdown', 'value'),
         State('document-text', 'value'),
         State('type', 'value'),
         State('num-archetype', 'value')]
    )
    def match_document(clicked, corpus_id, document_text, type, num_archetype):
        if not document_text:
            raise PreventUpdate
        similarities, archetype_maps = analyze_text(
            corpus_id, document_text, type, num_archetype
        )

        fig = ff.create_annotated_heatmap(
            z=similarities.values.tolist(),
            y=similarities.index.tolist(),
            x=similarities.columns.tolist(),
            xgap=1,
            ygap=1,
            colorscale='Viridis',
            colorbar={"title": "Percentage"},
            showscale=True
        )

        fig['layout']['title'] = 'Document Archetype Matching'
        fig['layout']['yaxis']['title'] = 'Archetype'

        cols = 2
        maxrows = int(1 + num_archetype//cols)
        fig2 = make_subplots(
            rows=maxrows,
            cols=cols,
            horizontal_spacing=.2,
            subplot_titles=[
                'Archetype {} vs DOC'.format(i) for i in range(num_archetype)
            ]
        )
        for i in range(num_archetype):
            fig2.add_trace(
                go.Heatmap(
                    z=archetype_maps[i].values.tolist(),
                    y=archetype_maps[i].index.tolist(),
                    x=[
                        'Archetype {}'.format(
                            archetype_maps[i].columns.tolist()[0]),
                        'DOC'
                    ],
                    xgap=1,
                    ygap=1,
                    name='Comparison'
                ), col=(i % cols + 1), row=(int(i // cols) + 1)
            )
            fig2.update_yaxes(
                tickangle=-30,
                tickfont={'size': 9},
                col=(i % cols + 1),
                row=(int(i // cols) + 1)
            )
        fig2.update_layout(
            height=400*maxrows,
            title_text='Archetype vs Document Comparisons',
        )

        return fig, fig2
