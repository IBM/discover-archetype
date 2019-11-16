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
