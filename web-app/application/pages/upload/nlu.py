import os
import pickle
import time

from ibm_watson import NaturalLanguageUnderstandingV1 as NaLaUn
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import \
    Features, CategoriesOptions, \
    ConceptsOptions, EntitiesOptions, KeywordsOptions, \
    RelationsOptions, SyntaxOptions

from application.models import db, Corpus, CorpusResult


def analyze_corpus(app, name, directory):
    features = Features(
        categories=CategoriesOptions(),
        concepts=ConceptsOptions(),
        entities=EntitiesOptions(),
        keywords=KeywordsOptions(),
        relations=RelationsOptions(),
        syntax=SyntaxOptions()
    )
    with app.app_context():
        authenticator = IAMAuthenticator(
            app.config['NATURAL_LANGUAGE_UNDERSTANDING_IAM_APIKEY']
        )
        service = NaLaUn(
            version=app.config['NATURAL_LANGUAGE_UNDERSTANDING_VERSION'],
            authenticator=authenticator)
        service.set_service_url(
            app.config['NATURAL_LANGUAGE_UNDERSTANDING_URL']
        )

        filenames = os.listdir(directory)
        new_corpus = Corpus(name=name, status='processing')
        db.session.add(new_corpus)
        db.session.commit()
        db.session.flush()
        print('Analyzing corpus in thread. Corpus ID: ' + str(new_corpus.id))
        count = 0
        for file in filenames:
            path = os.path.join(directory, file)
            if not os.path.isfile(path) or not file.endswith('.txt'):
                continue
            with open(path) as f:
                for i in range(3):
                    try:
                        results = service.analyze(
                            text=f.read(),
                            features=features
                        )
                        pickled_results = pickle.dumps(results)
                        new_results = CorpusResult(
                            corpus_id=new_corpus.id,
                            name=file.replace('.txt', ''),
                            data=pickled_results)
                        db.session.add(new_results)
                        db.session.commit()
                        count += 1
                        print('Processed file #{}: {} '.format(count, file))
                    except Exception as e:
                        print(e)
                        time.sleep(0.5)
                        print('Retrying...')
                    else:
                        break
                else:
                    print('Failed to analyze a file ({}) after ' +
                          'multiple attempts.'.format(file))

        new_corpus.status = 'ready'
        db.session.commit()
        print('Finished analyzing corpus.')
