import base64
import os
import shutil
import threading
import zipfile

from dash.dependencies import Input, Output, State
from flask import current_app

from .nlu import analyze_corpus
from application.models import db, Corpus, CorpusResult


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

    extract_directory = os.path.join(
        UPLOAD_FOLDER, filename.replace('.zip', '')
    )
    if os.path.exists(extract_directory):
        shutil.rmtree(extract_directory)
    os.makedirs(extract_directory)
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(extract_directory)
    os.unlink(zip_path)
    return extract_directory


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
        extract_directory = parse_contents(content, filename)
        thread = threading.Thread(
            target=analyze_corpus,
            args=(current_app._get_current_object(),
                  filename.replace('.zip', ''),
                  extract_directory)
        )
        thread.daemon = True
        thread.start()
        # analyze_corpus(filename.replace('.zip',''), extract_directory)
        return 'Corpus successfully uploaded and is currently being processed.'

    @dash_app.callback(
        Output('datatable-paging', 'data'),
        [Input('datatable-paging', 'page_current'),
         Input('datatable-paging', 'page_size'),
         Input('output-data-upload', 'children'),
         Input('delete-output', 'children')]
    )
    def update_table(page_current, page_size, upload_output, delete_output):
        results = Corpus.query.all()
        results_list = []
        for result in results[::-1]:
            results_list.append({
                'id': result.id,
                'corpus': result.name,
                'status': result.status
            })
        return results_list[page_current*page_size:(page_current+1)*page_size]

    @dash_app.callback(
        Output('delete-output', 'children'),
        [Input('delete-button', 'n_clicks')],
        [State('delete-input', 'value')]
    )
    def update_deletion_output(n_clicks, value):
        if value is None:
            return ''
        result = Corpus.query.get(value)
        if not result:
            return 'Corpus with ID {} does not exist.'.format(value)
        delete_q = CorpusResult.__table__.delete().where(
            CorpusResult.corpus_id == value
        )
        delete_q2 = Corpus.__table__.delete().where(Corpus.id == value)
        db.session.execute(delete_q)
        db.session.execute(delete_q2)
        db.session.commit()
        return 'Corpus with ID {} successfully deleted.'.format(value)
