from flask import current_app
import pandas as pd

from ibm_watson import NaturalLanguageUnderstandingV1 as NaLaUn
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import \
    Features, ConceptsOptions, EntitiesOptions, KeywordsOptions

from application.analysis.archetypes import create_archetypes, norm_dot
from application.analysis.corpus import get_corpus_results


def get_corpus_archetypes(corpus_id, type='concepts', n_archs=6):
    df_dic = get_corpus_results(corpus_id)
    archetypes = create_archetypes(corpus_id, type, n_archs, df_dic)
    return archetypes


def analyze_text(corpus_id, text, type, n_archs):
    features = Features(
        concepts=ConceptsOptions(),
        entities=EntitiesOptions(),
        keywords=KeywordsOptions(),
    )
    authenticator = IAMAuthenticator(
        current_app.config['NATURAL_LANGUAGE_UNDERSTANDING_IAM_APIKEY']
    )
    service = NaLaUn(
        version=current_app.config['NATURAL_LANGUAGE_UNDERSTANDING_VERSION'],
        authenticator=authenticator)
    service.set_service_url(
        current_app.config['NATURAL_LANGUAGE_UNDERSTANDING_URL']
    )
    response = service.analyze(
        text=text,
        features=features
    )
    results = {}
    typ_list = ['entities', 'concepts', 'keywords']
    for typ in typ_list:
        results[typ] = pd.DataFrame(response.result[typ])

    test_vec = \
        results['concepts'].set_index('text')[['relevance']].apply(norm_dot)
    print(test_vec)
    archetypes = get_corpus_archetypes(corpus_id, type=type, n_archs=n_archs)

    # Select the subset of features in corpus that cover the test vector.
    in_common = list(set(test_vec.index).intersection(
        set(archetypes.fn.columns)
    ))
    print(in_common)

    similarities = (
        (archetypes.fn[in_common] @ test_vec.loc[in_common]) * 100
    ).applymap(int)
    similarities.columns = ['similarity %']
    return similarities
