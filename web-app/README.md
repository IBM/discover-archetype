# Archetype Discovery Web App

A browser based UI with 3 tabs:
  1. Run Watson NLU on a new corpus and save the results.
  2. Compute the archetypes and analyze them.
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


## Run application

To run the application in development mode to try things out, run the following commands:

```bash
$ export FLASK_APP=app.py
$ flask run
```

This will deploy a development server which can be viewed in a browser by visiting
`http://127.0.0.1:5000/`.
