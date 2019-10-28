import dash_core_components as dcc
import dash_html_components as html


layout = html.Div([
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
], className='mt-4')
