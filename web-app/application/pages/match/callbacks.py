from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.figure_factory as ff

from .watson import analyze_text


def register_callbacks(dash_app):

    @dash_app.callback(
        Output('output-graph', 'figure'),
        [Input('submit-button', 'n_clicks')],
        [State('corpus-dropdown', 'value'),
         State('document-text', 'value'),
         State('type', 'value'),
         State('num-archetype', 'value')]
    )
    def match_document(clicked, corpus_id, document_text, type, num_archetype):
        if not document_text:
            raise PreventUpdate
        similarities = analyze_text(
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
        return fig
