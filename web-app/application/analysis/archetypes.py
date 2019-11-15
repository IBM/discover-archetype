import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


# This will act as our cache for already created Archetypes objects.
archetypes_dic = {}


def norm_dot(vec, weights=False):
    '''
    Normalizes a vector - dot product: v @ v = 1
    '''
    if weights:
        return np.sqrt(vec @ vec)

    return vec / np.sqrt(vec @ vec)


def norm_sum(vec, weights=False):
    '''
    Normalizes a vector - sum: v.sum = 1
    '''
    if weights:
        return vec.sum()

    return vec / vec.sum()


def scale(vec, weights=False):
    '''
    Normalizes a vector: v.min = 0, v.max = 1
    '''
    stop_divide_by_zero = 0.00000001
    if weights:
        return (vec.max()-vec.min() + stop_divide_by_zero)
    return (vec-vec.min())/(vec.max()-vec.min() + stop_divide_by_zero)


def create_archetypes(corpus_id, typ='entities', n_archs=6, df_dic={}):
    if corpus_id not in archetypes_dic.keys():
        archetypes_dic[corpus_id] = {}
    if typ not in archetypes_dic[corpus_id].keys():
        archetypes_dic[corpus_id][typ] = {}
    if n_archs not in archetypes_dic[corpus_id][typ].keys():
        archetypes_dic[corpus_id][typ][n_archs] = {}
        df = pd.DataFrame()
        for key in df_dic:
            dfx = df_dic[key][typ].copy()
            dfx['dictation'] = key
            df = df.append(dfx, sort=True)
        if typ == 'entities':
            df = df[df['type'] == 'HealthCondition']
            df.rename({'relevance': 'rel0'}, axis=1, inplace=True)
            df['relevance'] = df['rel0'] * df['confidence']
        mat = df.pivot_table(
            index='dictation', columns='text', values='relevance'
        ).fillna(0)
        archetypes_dic[corpus_id][typ][n_archs] = Archetypes(mat, n_archs)
    return archetypes_dic[corpus_id][typ][n_archs]


class Archetypes:
    '''
    Archetypes: Performs NMF of order n on X and stores the result as
                attributes.
    Archetypes are normalized: cosine similarity a(i) @ a(i) = 1.
    Atributes:
        my_archetypes.n         - order / number of archetypes
        my_archetypes.X         - input matrix

        my_archetypes.model     - NMF model
        my_archetypes.w         - NMF w-matrix
        my_archetypes.h         - NMF h-matrix

        my_archetypes.o         - occupations x archetypes matrix
                                 (from w-matrix)
        my_archetypes.on        - occupations x normalized archetypes matrix
                                 (from w-matrix) - SOCP number as index.
        my_archetypes.occ       - occupations x normalized archetypes matrix -
                                  Occupation names as index

        my_archetypes.f         - features x archetypes matrix (from h-matrix)
        my_archetypes.fn        - features x normalized archetypes matrix

    '''
    def __init__(self, X, n, norm=norm_dot):
        self.n = n
        self.X = X

        self.model = NMF(n_components=n, init='random', random_state=0,
                         max_iter=1000, tol=0.0000001)
        self.w = self.model.fit_transform(self.X)
        self.h = self.model.components_

        self.o = pd.DataFrame(self.w, index=self.X.index)
        self.on = self.o.T.apply(norm).T
        self.occ = self.on.copy()

        self.occ['Occupations'] = self.occ.index
        self.occ = self.occ.set_index('Occupations')

        self.f = pd.DataFrame(self.h, columns=X.columns)
        self.fn = self.f.T.apply(norm).T
