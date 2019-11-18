

## We emulate a new transcript by picking one from our test set. Not included in the corpus. 

test_name = wda.names_test[1]

test_text = wda.dictation_df.loc[test_name]
test_text

## DAVID TO TEAM: See self.watson and self.watson_nlu - which does the Watson analysis 
## for all documents in the corups. Reuse code?.



# ## Call Watson API
# def watson_nlu(text, 
#                typ_list = ['entities','concepts','keywords']):
#     module = wda.nlu_model.analyze(text = text, features=NLU['features'])
#     result = {}
#     for typ in typ_list:
#          result[typ] = pd.DataFrame(module.result[typ])
#     return result

# test_watson = watson_nlu(test_text)

test_watson = wda.watson_nlu[test_name]
test_watson
    

test_watson['concepts']


## Construct the 'concepts'-word vector

test_vec = test_watson['concepts'].set_index('text')[['relevance']].apply(norm_dot)
test_vec


# MAP TEST DOCUMENT ON ARCHETYPES
archetypes = wda.archetypes(typ='concepts',n_archs=6)

archetypes.f


## Select the subset of features in corpus that cover the test vector.
in_common     = list(set(test_vec.index).intersection(set(archetypes.fn.columns)))

## Check if the test vector contains new features that are not in corpus
beyond_corpus = list(set(test_vec.index) - set(archetypes.fn.columns))

## Display
in_common, beyond_corpus


# Measure the similarities between the test vector and the archetypes
sns.set(font_scale = 2)
similarities = ((archetypes.fn[in_common] @ test_vec) * 100).applymap(int)
similarities.columns = ['similarity %']
plt.figure(figsize = (2,6))
sns.heatmap(similarities,annot=True)
plt.ylabel('Archetype')
plt.title('MY DOC match with ALL Archetypes-features (%)')


scale_segment = np.sqrt(archetypes.fn.shape[1]/len(test_vec))
sns.set(font_scale = 2)
similarities = ((archetypes.fn[in_common]* scale_segment @ test_vec) * 100).applymap(int)
similarities.columns = ['similarity %']
plt.figure(figsize = (2,6))
sns.heatmap(similarities,annot=True)
plt.ylabel('Archetype')
plt.title('Archetypes match with MY DOC-feature subset (%)')


sns.set(font_scale = 2)
compare = archetypes.fn[in_common].T
# mx = np.sqrt(archetypes.fn.shape[1]/len(test_vec))
mx = 1
compare = compare * mx
compare['MY DOC'] = test_vec.loc[in_common].apply(scale)
compare = compare.sort_values(by='MY DOC', ascending = False)[['MY DOC']+list(compare.columns)[:-1]]
plt.figure(figsize = (6,6))
sns.heatmap((compare*100).applymap(np.sqrt),linewidths = 1,cbar=False)
plt.xlabel('Archetypes')
plt.title('Match with Archetypes')


test_vec_expanded = pd.DataFrame(test_vec, index = archetypes.f.columns).apply(scale).fillna(-0.1)
test_vec_expanded.min()

test_vec_expanded = pd.DataFrame(test_vec, index = archetypes.f.columns).apply(scale).fillna(0)

sns.set(font_scale = 1.5)
compare = archetypes.f.T.apply(scale)
compare['MY DOC'] = test_vec_expanded.apply(scale)
for ix in archetypes.f.index:
    cmp = compare.sort_values(by=ix,ascending=False)[[ix,'MY DOC']]
    cmp = cmp[cmp[ix] >0.1]
    plt.figure(figsize = (2,6))
    sns.heatmap(cmp.applymap(np.sqrt),linewidth = 1)
    plt.show()

      
      
      