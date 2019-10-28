import base64
import os
import zipfile

from dash.dependencies import Input, Output, State


UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = set(['zip'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_contents(content, filename):
    content_type, content_string = content.split(',')
    zip_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    with open(zip_path, "wb") as fh:
        fh.write(base64.b64decode(content_string))

    extract_directory = os.path.join(UPLOAD_FOLDER, 'extracted')
    if not os.path.exists(extract_directory):
        os.makedirs(extract_directory)
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(extract_directory)


def register_callbacks(dash_app):

    @dash_app.callback(
        Output('output-data-upload', 'children'),
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def update_output(content, filename):
        if not content:
            return ''

        if not allowed_file(filename):
            return 'Uploaded file is not a zip file.'
        print('in update callback')
        parse_contents(content, filename)
        return []
