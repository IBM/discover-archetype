import pickle

from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from application.models import Corpus, CorpusResult
from .archetypes import display_archetype


def register_callbacks(dash_app):

    def get_corpus_results(corpus_id):
        results = CorpusResult.query.filter(
            CorpusResult.corpus_id == corpus_id
        ).all()

        df_dic = {}
        for result in results:
            watson_response = pickle.loads(result.data)
            df_dic[result.name] = {}
            for item in list(watson_response.result.items()):
                df_dic[result.name][item[0]] = pd.DataFrame(list(item[1]))

        return df_dic

    @dash_app.callback(
        Output('variables-heatmap', 'figure'),
        [Input('Var', 'value'),
         Input('NoA', 'value'),
         Input('Threshold', 'value'),
         Input('corpus-dropdown', 'value')]
    )
    def arch_heatmap_variables(typ, n_archs, threshold, corpus_id):
        print('--- In Heatmap Dash callback ---')
        df_dic = get_corpus_results(corpus_id)

        def f(i):
            # Display and sort by archetype i
            return display_archetype(
                corpus_id, arch_nr=i, typ=typ, n_archs=n_archs,
                threshold=threshold, df_dic=df_dic
            ).sort_values(by=i)

        maxrows = int(1 + n_archs//3)
        cols = 3
        fig = make_subplots(rows=maxrows, cols=cols, horizontal_spacing=.2)
        for i in range(n_archs):
            res = f(i)
            fig.add_trace(
                go.Heatmap(
                    z=res.values.tolist(),
                    y=res.index.tolist(),
                    x=res.columns.tolist(),
                    xgap=1,
                    ygap=1,
                ), col=(i % cols + 1), row=(int(i // cols) + 1)
            )
        fig.update_layout(
            height=400*maxrows,
            width=1100,
            title_text="Discovered Archetypes"
        )
        return fig

    @dash_app.callback(
        Output('corpus-dropdown2', 'options'),
        [Input('corpus-dropdown2', 'search_value')],
    )
    def update_corpus_options(search_value):
        print('in update corpus options')

        options = []
        corpora = Corpus.query.all()
        for corpus in corpora:
            options.append({
                'label': '{} (ID: {})'.format(corpus.name, corpus.id),
                'value': corpus.id
            })
        return options
