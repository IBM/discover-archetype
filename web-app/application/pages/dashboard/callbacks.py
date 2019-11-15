from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from application.analysis.archetypes import create_archetypes
from application.analysis.corpus import get_corpus_results


def register_callbacks(dash_app):

    def display_archetype(corpus_id, typ='entities', n_archs=6, arch_nr=0,
                          threshold=0.1, df_dic={}):
        arc = create_archetypes(
            corpus_id, typ, n_archs, df_dic).f.T.sort_values(
                by=arch_nr, ascending=False
        )
        return arc[arc[arch_nr] >= (threshold * arc[arch_nr][0])]

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

        cols = 2
        maxrows = int(1 + n_archs//cols)
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
            fig.update_yaxes(
                tickangle=-30,
                tickfont={'size': 9},
                col=(i % cols + 1),
                row=(int(i // cols) + 1)
            )
        fig.update_layout(
            height=400*maxrows,
            width=1100,
            title_text="Discovered Archetypes"
        )
        return fig
