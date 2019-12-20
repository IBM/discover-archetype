import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def layout():
    return html.Div([
        html.H3('Upload Corpus for Archetype Discovery'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File', href='')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
            # Allow multiple files to be uploaded
            multiple=False,
            accept='.zip',
            className='mt-4'
        ),
        html.Div(id='output-data-upload'),
        html.Div([
            html.H3('Uploaded Corpora'),
            dash_table.DataTable(
                id='datatable-paging',
                columns=[
                    {'name': 'ID', 'id': 'id'},
                    {'name': 'Corpus Name', 'id': 'corpus'},
                    {'name': 'Status', 'id': 'status'}
                ],
                page_current=0,
                page_size=20,
                page_action='custom'
            )
        ], style={'margin-top': '50px'}),
        html.Div([
            html.H4('Corpus Management'),
            html.Div([
                dbc.Row(
                    dbc.Col([
                        html.P('Corpus Deletion',
                               style={'font-weight': 'bold'}),
                        html.P('Enter the ID of the corpus you want '
                               'to delete.'),
                        dbc.Form(
                            [
                                dbc.FormGroup([
                                    dbc.Input(id='delete-input',
                                              type='text',
                                              placeholder='Enter Corpus ID'),
                                ], className='mr-3 mb-0'),
                                dbc.Button('Delete',
                                           color='danger',
                                           id='delete-button'),
                            ],
                            inline=True,
                        ),
                        html.Div(id='delete-output',
                                 children='')
                    ], width=12)
                )
            ], style={'background': '#ececec', 'padding': '10px'})
        ], style={'margin-top': '50px'}),
    ], className='mt-4 mb-4')
