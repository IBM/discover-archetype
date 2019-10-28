import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


def norm_dot(vec, weights=False):
    '''
    Normalizes a vector - dot product: v @ v = 1
    '''
    if weights:
        return np.sqrt(vec @ vec)

    return vec / np.sqrt(vec @ vec)


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


archetypes_dic = {}


def archetypes(typ='entities', n_archs=6, df_dic={}):
    if typ not in archetypes_dic.keys():
        archetypes_dic[typ] = {}
    if n_archs not in archetypes_dic[typ].keys():
        archetypes_dic[typ][n_archs] = {}
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
        archetypes_dic[typ][n_archs] = Archetypes(mat, n_archs)
    return archetypes_dic[typ][n_archs]


def display_archetype(typ='entities', n_archs=6, arch_nr=0, var='variables',
                      threshold=0.1, df_dic={}):
    if var == 'variables':
        arc = archetypes(typ, n_archs, df_dic).f.T.sort_values(
            by=arch_nr, ascending=False
        )
        result = arc[
            arc[arch_nr] >= (threshold * arc[arch_nr][0])
        ]
        return result
