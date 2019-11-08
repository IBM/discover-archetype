from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .watson import analyze_text


def register_callbacks(dash_app):

    @dash_app.callback(
        Output('output', 'children'),
        [Input('submit-button', 'n_clicks')],
        [State('corpus-dropdown', 'value'),
         State('document-text', 'value'),
         State('type', 'value'),
         State('num-archetype', 'value')]
    )
    def match_document(clicked, corpus_id, document_text, type, num_archetype):
        if not document_text:
            raise PreventUpdate
        print('in match document')
        print(corpus_id)
        similarities = analyze_text(
            corpus_id, document_text, type, num_archetype
        )
        print(similarities)
