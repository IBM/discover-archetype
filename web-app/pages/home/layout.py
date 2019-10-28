import dash_html_components as html


layout = html.Div([
    html.H1('Discover Archetypes'),
    html.P('''
        In any corpus of unstructured data from an arbitrary domain, there are
        usually clusters of co-occuring features that the brain understands
        as topics or archetypes of that domain, which it often uses to
        characterized and label the domain. For instance, in a database of
        medical dictations, the language will obviously contain a larger
        than average proportion of medical words, which will co-occur in
        patterns that represent the medical conditions of the patients
        (as understood by the physicians behind the dictations). These
        clusters can be seen as topics, or archetypes of medical
        conditions, depending on how we choose to frame them.
    '''),
    html.P('''
        In this web app, you can (1) run Watson NLU on a new corpus and save
        the results, (2) compute the archetypes and analyze them, and (3)
        match a new documents with the archetypes and see the relevant terms.
    ''')
], className='mt-4')
