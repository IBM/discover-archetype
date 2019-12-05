#++++++++++++++++++++++++++++++++++++++++++++++
# Before running the script, edit 
# 'SET HYPERPARAMETERS' 
# - the rest is automated
#++++++++++++++++++++++++++++++++++++++++++++++

##################
### IMPORT MODULES
##################

### System
import sys
import os
from fnmatch import fnmatch 

### I/O
import json
import pickle

### General Processing
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import math
import collections
from collections import OrderedDict
from collections import namedtuple
DottedDict = namedtuple

## DECOMPOSITION
from sklearn.decomposition import NMF
from scipy.linalg import svd
from sklearn.model_selection import train_test_split

### NLU
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1 as NLUV1
#from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions,ConceptsOptions,EntitiesOptions,KeywordsOptions,RelationsOptions,SyntaxOptions


### Presentation / apps
from matplotlib import pyplot as plt
import seaborn as sns

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

## GENERAL FUNCTIONS 

### SELECTION
def random_split(lst,split=0.5):
    shuffled = np.array(lst)
    np.random.shuffle(shuffled)
    split    = int(split * len(shuffled))
    return  shuffled[-split:] , shuffled[:-split]

### NORMALIZATION
#### Statistic normalization - subtract mean, scale by standard deviation
def norm_stat(vec, weights = False):
    '''
    Normalizes a vector v-v.mean())/v.std() 
    '''
    if weights:
        return  np.mean(abs(vec - vec.mean()))  
    
    return (vec-vec.mean())/vec.std()

#### Algebraic normalization - dot product
def norm_dot(vec, weights = False):
    '''
    Normalizes a vector - dot product: v @ v = 1
    '''
    if weights:
        return  np.sqrt(vec @ vec)
    
    return vec / np.sqrt(vec @ vec)

#### Algebraic normalization - dot product
def norm_sum(vec, weights = False):
    '''
    Normalizes a vector - sum: v.sum = 1
    '''
    if weights:
        return  vec.sum()
    
    return vec / vec.sum()

#### Scaled Normalization -
def scale(vec, weights = False):
    '''
    Normalizes a vector: v.min = 0, v.max = 1
    '''
    stop_divide_by_zero = 0.00000001
    if weights:
        return (vec.max()-vec.min() + stop_divide_by_zero)
    return (vec-vec.min())/(vec.max()-vec.min() + stop_divide_by_zero)
def cleanup_chars(string,char_list = ('\n',' ')):
    result = string
    for char in char_list:
        result = result.replace(char,'')
    return result

### Matrix operations

def dotdf(df1,df2):
    '''
    performs df1 @ df2 without exceptions, when df1.columns and df2.index are not identical
    '''
    c = set(df1.columns)
    i = set(df2.index)
    var = list(c - (c - i))
    return df1[var] @ df2.loc[var]

### OS system commands

from fnmatch import fnmatch 
def ls(search,name_only = False,cos=None):
    '''
    emulates unix ls (without flags). Accepts wildcard/'*' in 
    '''
    search_split = search.replace('/','/ ').split()
    pattern      =         search_split[ -1]
    path         = ''.join(search_split[:-1])
    if cos is None:
        # look in filesystem
        all_names = np.array(os.listdir(path)) # numpy array enables Boolean Mask
    else:
        # look in cloud object store
        all_names = np.array(cos.get_bucket_contents())
    if not name_only and cos is None: # add path to each name
        all_names    = np.array([path+name for name in all_names]) 
    mask         = [fnmatch(name,pattern) for name in all_names]
    result       = all_names[mask]
    return result

### plotting function

def plot_archetypes(wda, feature_type):
    # Create a plot for the number of archetypes desired 
    fig = plt.figure(figsize=(14, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n_archs   = 6
    threshold = 0.1
    for i in range(n_archs):
        tot = wda.archetypes(typ=feature_type, n_archs=n_archs).f.T.apply(scale)[i].sum()
        f =   wda.display_archetype(typ=feature_type, n_archs=n_archs, arch_nr=i, threshold=threshold, norm = scale)
        ax = fig.add_subplot(2, 3, i+1)
        ax.title.set_text('Key: Archetype '+str(i)+'\n'+str(int(100*f[i].sum()/tot))+'% of volume displayed')
        sns.heatmap(f)
    return fig

def plot_coordinate(wda, feature_type):
    # Plot as a coordinate system
    n_archs = 6
    f = wda.archetypes(typ=feature_type,n_archs=n_archs).o.applymap(lambda X: X+0.00000000000000001).T
    sns.clustermap(f.apply(norm_sum).T)
    
def plot_similarity(wda, ):

# # MATRIX-FACTORIZATION: DIMENSIONALITY REDUCTION & ARCHETYPING

# ## CLUSTER FEATURES INTO OCCUPATION CATEGORIES
# ## Use non-zero matrix factorization for clustering
# ## Use singular value decomposition first state for determining overall similarity

class Archetypes:
    '''
    Archetypes: Performs NMF of order n on X and stores the result as attributes. 
    Archetypes are normalized: cosine similarity a(i) @ a(i) = 1. 
    Atributes:
        my_archetypes.n         - order / number of archetypes
        my_archetypes.X         - input matrix
        
        my_archetypes.model     - NMF model 
        my_archetypes.w         - NMF w-matrix 
        my_archetypes.h         - NMF h-matrix
              
        my_archetypes.f         - features x archetypes matrix (from h-matrix)
        my_archetypes.fn        - Dot-Normalized archetypes  
        
        my_archetypes.o         - documents x archetypes matrix (from w-matrix) 
        my_archetypes.on        - Sum-Normalized documents
        
    '''
    
    def __init__(self,X,n,
                 norm         = norm_dot,
                 bootstrap    = False,     bootstrap_frac = 0.5,
                 random_state = None):
        self.n = n
        self.X = X
        self.norm = norm
        self.random_state = random_state
        
        if bootstrap:
            self.bootstrap_n    = bootstrap
            self.bootstrap_frac = bootstrap_frac
        else:
            self.bootstrap_n    = 1
            self.bootstrap_frac = 1
        
        self.model = NMF(n_components=n, init='random', random_state=self.random_state, max_iter = 1000, tol = 0.0000001)
        self.w_dic = {}
        self.o_dic = {}
        self.h_dic = {}
        self.f_dic = {}   
        
        for j in range(self.bootstrap_n): 
            XX = self.X.sample(int(len(self.X) *self.bootstrap_frac))
            self.w_dic[j] = self.model.fit_transform(XX)
            self.o_dic[j] = pd.DataFrame(self.w_dic[j],index=XX.index)
            self.h_dic[j] = self.model.components_
            self.f_dic[j] = pd.DataFrame(self.h_dic[j],columns=XX.columns)
        
        self.w  = self.w_dic[0]  # TEMPORARY
        self.o  = self.o_dic[0]  # TEMPORARY
        self.h  = self.h_dic[0]  # TEMPORARY
        self.f  = self.f_dic[0]  # TEMPORARY
 
        
        self.fn          = self.f.T.apply(norm_dot).T    # under construction - David
        self.on          = self.o.T.apply(norm_sum).T
        self.norm_matrix = self.fn @ self.fn.T           # under construction - David
        self.arch_basis  = self.norm_matrix.T @ self.fn  # under construction - David


class Svd:
    ''''
    Singular value decomposition-as-an-object
        my_svd = Svd(X) returns
        my_svd.u/.s/.vt – U S and VT from the Singular Value Decomposition (see manual)
        my_svd.f        – Pandas.DataFrame: f=original features x svd_features
        my_svd.o        - Pandas.DataFrame: o=occupations x svd_features
        my_svd.volume(keep_volume) 
                        - collections.namedtuple ('dotted dicionary'): 
                          Dimensionality reduction. keeps 'keep_volume' of total variance
                          
                          
    '''
    def __init__(self,X,):
        self.u,self.s,self.vt = svd(np.array(X))
        self.f = pd.DataFrame(self.vt,columns=X.columns)
        self.o = pd.DataFrame(self.u,columns=X.index)
        
    def volume(self,keep_volume):
        ''' 
        Dimensionality reduction, keeps 'keep_volume' proportion of original variance
        Type: collections.namedtuple ('dotted dictionary')
        Examples of usage:
        my_svd.volume(0.9).s - np.array: eigenvalues for 90% variance 
        my_svd.volume(0.8).f - dataframe: features for 80% variance
        my_svd.volume(0.5).o - dataframe: occupations for 50% variance      
        '''
        dotted_dic = collections.namedtuple('dotted_dic', 's f o')
        a1 = self.s.cumsum()
        a2 = a1/a1[-1]
        n_max = np.argmin(np.square(a2 - keep_volume))
        cut_dic = dotted_dic(s= self.s[:n_max],f= self.f.iloc[:n_max], o= self.o.iloc[:n_max])
        return cut_dic
        


class WatsonDocumentArchetypes:
    '''
    WatsonDocumentArchetypes performs Archetypal Analysis on a corpus consisting of a set of documents, for example a set 
    of articles, books, news stories or medical dictations.
    
    Input parameters:
    
    PATH            - Dictionary with paths to I/O
    PATH['data']    - Directory for input text files. Example: './data/input_texts/'
    PATH['results'] - Directory for output.           Example: './data/output_nlu/'
    
    NLU                   - Dictionary with information for running Watson NLU
    NLU['apikey']         - apikey for running Watson NLU
    NLU['apiurl']         - URL for Watson NLU API
    NLU['version']        - Watson NLU version, e.g. '2019-07-12'
    NLU['features']       - Features requested from Watson NLU for each document in the set, e.g. 
                                Features(
                                categories= CategoriesOptions(),
                                concepts  = ConceptsOptions(),
                                entities  = EntitiesOptions(),
                                keywords  = KeywordsOptions(),
                                relations = RelationsOptions(),
                                syntax    = SyntaxOptions()
                                )

    Attributes:

        
        self.PATH 
    
        
    '''
    
    def __init__(self, PATH, NLU, 
                 train_test = False,
                 random_state = None,
                 use_cloud_store = False):
        
        from cloud_object_store import CloudObjectStore
        
        self.PATH         = PATH
        self.NLU          = NLU
        self.random_state = random_state      
        # To random partition documents into train/test-sets, 
        # choose relative size of test-set, train_test (1 = 100%)
        self.train_test = train_test  
        self.use_cloud_store = use_cloud_store
        
        # Create clients to interface Watson and Cloud services
        authenticator = IAMAuthenticator(NLU['apikey'])
        self.nlu_model  = NLUV1(version=NLU['version'], authenticator=authenticator)
        self.nlu_model.set_service_url(NLU['apiurl'])
        if self.use_cloud_store:
            self.cos_dictations=CloudObjectStore(PATH['dictation_bucket'], 
                                                PATH['cos_dictation_apikey'], 
                                                PATH['cos_dictation_crn'], 
                                                PATH['cos_dictation_endpoint'])
            self.cos_nlu=CloudObjectStore(PATH['nlu_bucket'], 
                                        PATH['cos_nlu_apikey'], 
                                        PATH['cos_nlu_crn'], 
                                        PATH['cos_nlu_endpoint'])
        
            
            # Initiate X_matrix dictionaries
        self.X_matrix_dic = {}
        self.X_matrix_train_dic = {}
        self.X_matrix_test_dic  = {}
        self.archetypes_dic = {} 
        self.svd_dic = {}
 
        ################
        ## PREPARE DATA 
        ################
        if self.use_cloud_store:
            # load from cloud storage bucket
            self.filenames = ls('*.txt', name_only=True, cos=self.cos_dictations)
        else:
            # load from local file system
            self.filenames = ls(self.PATH['data']+'*.txt', name_only=True)  # all filenames ending with '.txt' 
 
        self.names     = [name.replace('.txt','') for name in self.filenames]
        self.all_names = self.names *1      # if train_test - self.names will be set to self.names_train
        self.dictation_dic = {}             # dictionary for dictation files
        for name in self.filenames:
            if (self.use_cloud_store):
                self.dictation_dic[name.replace('.txt','')] = self.cos_dictations.get_item(name).decode('utf-8')
            else:
                self.dictation_dic[name.replace('.txt','')] = open(self.PATH['data']+name, encoding="utf-8").read()
        self.dictation_df = pd.Series(self.dictation_dic)
            
        ####################
        ## TRAIN-TEST SPLIT 
        ####################
        if self.train_test: # 0<train_test<1 - the proportion of names to save as 'test (rounded downwards)
            self.names_test , self.names_train = random_split(self.all_names , self.train_test)
            self.names = self.names_train

        ###############################
        ## PERFORM WATSON NLU ANALYSIS
        ###############################
        
        # QQQQQQQQQQQQQQQQQQ TODO QQQQQQQQQQQQQQQQQQ
        #   * IF DICTATION ALREADY HAS PKL WITH Watson NLU: READ EXISTING PKL. SKIP NEW WATSON CALC.
        #
 
        self.watson = {}    #Dictionary with Watson-NLU results for each dictation
        
        
        if self.use_cloud_store:
            # Check in Cloud storage bucket
            self.watson_pkl = 'all_dictations_nlu.pkl'
            pkl_exists = self.watson_pkl in self.cos_nlu.get_bucket_contents()
        else:
            # Check in local filesystem 
            self.watson_pkl = PATH['results']+'all_dictations_nlu.pkl'  
            pkl_exists = os.path.exists(self.watson_pkl)
 
        if pkl_exists:
            if self.use_cloud_store:
                # load previous result from Cloud storage
                self.watson = pickle.loads( self.cos_nlu.get_item(self.watson_pkl) )
            else:
                # load previous result from local filesystem
                self.watson = pickle.load( open( self.watson_pkl, "rb" ) )
                
        else: #perform nlu-analysis on dictations
            for item in list(self.dictation_dic.items()):
                lbl  = item[0]
                text = item[1]
                self.watson[lbl] = self.nlu_model.analyze(text = text, features=NLU['features'])
                if self.use_cloud_store:
                    # save result to Cloud storage
                    self.cos_nlu.create_item(str(lbl)+'_nlu.pkl', pickle.dumps(self.watson[lbl]))
                else:
                    # save result to local filesystem
                    f = open(PATH['results']+str(lbl)+'_nlu.pkl','wb')
                    pickle.dump(self.watson[lbl],f)
                    f.close()

            if self.use_cloud_store:
                # save result to Cloud storage
                self.cos_nlu.create_item(self.watson_pkl, pickle.dumps(self.watson))
            else:
                f = open(self.watson_pkl,'wb')
                pickle.dump(self.watson,f)
                f.close() 

        # Copy Watson NLU results to Pandas Dataframes
        self.watson_nlu = {}
        for dctn in self.watson.items():
            self.watson_nlu[dctn[0]] = {}
            for item in list(dctn[1].result.items()):
                self.watson_nlu[dctn[0]][item[0]]=pd.DataFrame(list(item[1]))


    ##############
    # ARCHETYPAL ANALYSIS
    ##############

    # CONSTRUCT X- MATRIX
    def X_matrix(self,typ = 'entities'):
        '''
        Construct the archetypal analysis X-matrix by pivoting the dataframe in the 
        dictionary my_wda.watson_nlu that contains the Watson NLU analysis in question
        
        X_matrix(typ)
            rows   : Dictations 
            columns: Variables; keywords/entities/concepts, from Watson NLU analysis
            values : Weights, from Watson NLU analysis
        
        the constructed X_matrix(typ) is saved as X_matrix_dic[typ]
        
        if my_wda.train_test has a value (not False) X_matrix_train_dic[typ] and X_matrix_test[typ]
        are added computed and added to their respective dicionaries
        '''
        if typ not in self.X_matrix_dic.keys():
            df = pd.DataFrame()
            for key in self.names:
                dfx = self.watson_nlu[key][typ].copy()
                dfx['dictation'] = key
                df = df.append(dfx,sort=True)
            if typ is 'entities':
                df = df[df['type']=='HealthCondition']
                df.rename({'relevance': 'rel0'}, axis=1,inplace=True)
                df['relevance'] = df['rel0'] * df['confidence']
            self.X_matrix_dic[typ] = df.pivot_table(index='dictation',columns='text',values='relevance').fillna(0)
        
        if self.train_test:
            self.X_matrix_train_dic[typ] = self.X_matrix_dic[typ]
            
            df = pd.DataFrame()
            for key in self.names_test:
                dfx = self.watson_nlu[key][typ].copy()
                dfx['dictation'] = key
                df = df.append(dfx,sort=True)
            if typ is 'entities':
                df = df[df['type']=='HealthCondition']
                df.rename({'relevance': 'rel0'}, axis=1,inplace=True)
                df['relevance'] = df['rel0'] * df['confidence']
            self.X_matrix_test_dic[typ] = df.pivot_table(index='dictation',columns='text',values='relevance').fillna(0)
        return self.X_matrix_dic[typ]

    # CALCULATE ARCHETYPES
    def archetypes(self,typ='entities',
                   n_archs=6,bootstrap = False, 
                   bootstrap_frac = 0.5, 
                   random_state = False,
                   norm = norm_sum):
        if random_state is False:
            random_state = self.random_state
        if typ not in self.archetypes_dic.keys():
            self.archetypes_dic[typ] = {}
        hyperparam = (n_archs,bootstrap,bootstrap_frac,random_state,norm)
        self.X_matrix(typ)
        self.archetypes_dic[typ][hyperparam] = Archetypes(self.X_matrix(typ),
                                                          n_archs,bootstrap = bootstrap, bootstrap_frac = bootstrap_frac,
                                                          random_state = random_state,
                                                          norm = norm)
        return self.archetypes_dic[typ][hyperparam]


    def display_archetype(self,arch_nr = -1, typ = 'entities' , n_archs = 6, var = 'variables', threshold = 0.10, norm = scale):
        fun = {'variables' : 'self.archetypes(typ = typ,n_archs = n_archs).f.T ',
               'dictations': 'self.archetypes(typ = typ,n_archs = n_archs).o'
               }
        f  = eval(fun[var])
        fn = f.apply(norm)
        if arch_nr == -1:
            return sns.clustermap(f).data2d
        else:
            arc        = fn.sort_values(by=arch_nr,ascending = False)
            if norm is norm_sum :  # normalized over sum: threshold is ignored volume
                arc_cs     = arc[arch_nr].cumsum() 
                thresh_idx = abs(arc_cs -(1- threshold)).values.argmin()
                result     = arc.iloc[:thresh_idx]
            if norm is scale :
                result = arc[
                            arc[arch_nr] >= (threshold * arc[arch_nr][0])                  ]
        return result

    
    # CALCULATE SVD
    def svd(self,typ='entities'):
        self.X_matrix(typ)
        self.svd_dic[typ] = Svd(self.X_matrix(typ))
        return 
    
    # ANALYZE A TEXT
    def analyze(self,text,typ='entities'):
        pass
    
