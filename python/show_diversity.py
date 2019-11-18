## David to team: as an extr feature, we can soncider includeing the dimensionality analyis - the plot - as a feature for analysis.


## INSTANTIATE THE WatsonDocumentArchetypes OBJECT as 'wda' 
# Split dictations into train/test-set with 5% set aside as test-dictations
# wda_a has no test-dictations, all are included in the corpus.

wda    = WatsonDocumentArchetypes(PATH,NLU,train_test = 0.05, random_state = 42)

wda_a  = WatsonDocumentArchetypes(PATH,NLU,train_test = False)


## DIMENSIONALITY OF THE CORPUS
# Establish with Singular Value Decomposition (principal component analysis) :

types = ['keywords','concepts','entities']
svde  = {}
volume = {}
volume_distribution = {}

for typ in types:
    svde[typ]     = Svd(wda.X_matrix(typ))
    volume[typ] = svde[typ].s.sum()
    volume_distribution[typ] = svde[typ].s.cumsum()/volume[typ]
    plt.plot(volume_distribution[typ],label = typ)
plt.title('DIMENSIONALITY OF WATSON DICTATION DATA ANALYSIS OUTPUT')
plt.xlabel('Dimensions / max = number of dictations in corpus')
plt.ylabel('Volume')
plt.legend()
plt.grid()
plt.show()

## COMMENT: CONCEPTS AND ENTITIES OFFER A GREATER REDUCTION OF DIMENSIONALITY = THE MODELS FIT THE DATA BETTER (?)
## ? => WE ARE MEASURING THE WATSON OUTPUT ONLY. WE DON'T KNOW WHAT IS GOING ON INSIDE WATSON. 

## CONCLUSIONS: 
## DICTATION WORD/ENTITY/CONCEPTUAL CONTENT IS DIVERSE AND SPREAD OVER MANY DIMENSIONS
## ACCESS TO A LARGER CORPUS SHOULD BE VERY GOOD!