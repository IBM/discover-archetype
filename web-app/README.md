# Archetype Discovery Web App

This web app showcases the archetype discovery process. Users can:
  1. Upload a corpus (`zip` file containing `txt` files) which will be processed by Watson NLU.
  2. Compute the archetypes of a corpus and analyze them.
  3. Match a new document with the archetypes and see the relevant terms.

## Install Dependencies

The general recommendation for Python development is to use a virtual environment
[(venv)](https://docs.python.org/3/tutorial/venv.html). To install and initialize a
virtual environment, use the `venv` module on Python 3.


```bash
$ python -m venv myvenv

# Now source the virtual environment. Use one of the two commands depending on your OS.

$ source mytestenv/bin/activate  # Mac or Linux
$ ./mytestenv/Scripts/activate   # Windows PowerShell
```

Next, in the project directory, run:

#### `pip install -r requirements.txt`


## Set Up Configuration

In the instance directory, there is a `config.py.sample` file. Copy this and fill in the necessary
credentials.

#### `cp instance/config.py.sample instance config.py`

By default, data will be stored in a locally created SQLite database file. However, you can use
whatever SQL database you desire as long you provide the correct database connection URI through the
`SQLALCHEMY_DATABASE_URI` option.

## Run application

To run the application in development mode to try things out, run the following command:

```bash
$ python app.py
```

This will deploy a development server which can be viewed in a browser by visiting
`http://127.0.0.1:5000/`.

Note: that the first start of the app might take a while, so please be patient.

### Uploading corpus

On the `/upload` page, users can upload a corpus for the Watson NLU service to process. This corpus
must be a `zip` file containing only `txt` files.

Uploaded corpora will be listed in a table also contained on the `/upload` page, and each corpus will
contain a status that will either be `ready` or `processing`. Users can check the console log from
where the Python app was started to get updates on the current processing. When an uploaded corpus has
completed processing by Watson NLU, the status in for the corpus in the table will change to `ready`
(page must be refreshed to see changes in status).

### Archetypes and Matching

After you have a corpus that has been uploaded and processed (in `ready` state), then you can then
select that corpus on the `/archetypes` and `/match' pages. With the select corpus, archetypes can be
extracted based on additional selected properties like the number of archetypes and the variable type
from Watson NLU (i.e., concepts, keywords, entities).
