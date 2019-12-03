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
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions,ConceptsOptions,EntitiesOptions,KeywordsOptions,RelationsOptions,SyntaxOptions


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


