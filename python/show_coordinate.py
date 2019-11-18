## ARCHETYPES based on ENTITIES in corpus texts
typ = 'entities'
n_archs = 6
f = wda.archetypes(typ=typ,n_archs=n_archs).o.applymap(lambda X: X+0.00000000000000001).T
sns.clustermap(f.apply(norm_sum).T)

## ARCHETYPES based on CONCEPTS in corpus texts
typ = 'concepts'
n_archs = 6
f = wda.archetypes(typ=typ,n_archs=n_archs).o.applymap(lambda X: X+0.00000000000000001).T
sns.clustermap(f.apply(norm_sum).T)


## ARCHETYPES based on KEYWORDS in corpus texts
typ = 'keywords'
n_archs = 6
f = wda.archetypes(typ=typ,n_archs=n_archs).o.applymap(lambda X: X+0.00000000000000001).T
sns.clustermap(f.apply(norm_sum).T)