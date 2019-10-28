import os
import pickle

from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .archetypes import display_archetype


def register_callbacks(dash_app):

    analyzed_directory = os.path.join(os.getcwd(), 'analyzed-results/')
    filename = 'all_dictations_nlu.pkl'
    dian = pickle.load(open(analyzed_directory + filename, "rb"))

    df_dic = {}
    for dctn in dian.items():
        df_dic[dctn[0]] = {}
        for item in list(dctn[1].result.items()):
            df_dic[dctn[0]][item[0]] = pd.DataFrame(list(item[1]))

    @dash_app.callback(
        Output('variables-heatmap', 'figure'),
        [Input('Var', 'value'),
         Input('NoA', 'value'),
         Input('Threshold', 'value')]
    )
    def arch_heatmap_variables(typ, n_archs, threshold):
        print('--- In Heatmap Dash callback ---')

        def f(i):
            # Display and sort by archetype i
            return display_archetype(
                arch_nr=i, typ=typ, n_archs=n_archs,
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
