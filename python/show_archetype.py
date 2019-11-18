

# From entities
typ = 'entities'
fig = plt.figure(figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
n_archs   = 6
threshold = 0.1
for i in range(n_archs):
    tot = wda.archetypes(typ = typ,n_archs=n_archs).f.T.apply(scale)[i].sum()
    f =   wda.display_archetype(typ = typ,n_archs=n_archs, arch_nr=i, threshold = threshold, norm = scale)
    ax = fig.add_subplot(2, 3, i+1)
    ax.title.set_text('Key: Archetype '+str(i)+'\n'+str(int(100*f[i].sum()/tot))+'% of volume displayed')
    sns.heatmap(f)
fig.show()

# From concepts
typ = 'concepts'
fig = plt.figure(figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
n_archs   = 6
threshold = 0.1
for i in range(n_archs):
    tot = wda.archetypes(typ = typ,n_archs=n_archs).f.T.apply(scale)[i].sum()
    f =   wda.display_archetype(typ = typ,n_archs=n_archs, arch_nr=i, threshold = threshold, norm = scale)
    ax = fig.add_subplot(2, 3, i+1)
    ax.title.set_text('Key: Archetype '+str(i+1)+'\n'+str(int(100*f[i].sum()/tot))+'% of volume displayed')
    sns.heatmap(f)
fig.show()

# From keywords
typ = 'keywords'
fig = plt.figure(figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
n_archs   = 6
threshold = 0.1
for i in range(n_archs):
    tot = wda.archetypes(typ = typ,n_archs=n_archs).f.T.apply(scale)[i].sum()
    f =   wda.display_archetype(typ = typ,n_archs=n_archs, arch_nr=i, threshold = threshold, norm = scale)
    ax = fig.add_subplot(2, 3, i+1)
    ax.title.set_text('Key: Archetype '+str(i+1)+'\n'+str(int(100*f[i].sum()/tot))+'% of volume displayed')
    sns.heatmap(f)
fig.show()

# From 