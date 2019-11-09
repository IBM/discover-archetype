import pickle

import pandas as pd

from application.models import CorpusResult


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
