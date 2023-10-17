# Vantage6 algorithm for logistic regression

This algorithm was designed for the [vantage6](https://vantage6.ai/) 
architecture. 

## Input data

The algorithm expects each data node to hold data that adheres to the same 
standard.

## Using the algorithm

Below you can see an example of how to run the algorithm:

``` python
import time
from vantage6.client import Client

# Initialise the client
client = Client('http://127.0.0.1', 5000, '/api')
client.authenticate('username', 'password')
client.setup_encryption(None)

# Define algorithm input
input_ = {
    'method': 'master',
    'master': True,
    'kwargs': {
        'org_ids': [2, 3],          # organisations to run algorithm
        'predictors': ['c1', 'c2'], # columns to be used as predictors
        'outcome': 'outcome',       # column to be used as outcome
        'classes': [0, 1],          # classes to be predicted
        'max_iter': 15,             # maximum number of iterations to perform
        'delta': 0.01,              # threshold loss difference for convergence
    }
}

# Send the task to the central server
task = client.task.create(
    collaboration=1,
    organizations=[2, 3],
    name='v6-logistic-regression-py',
    image='ghcr.io/maastrichtu-cds/v6-logistic-regression-py:latest',
    description='run logistic regression',
    input=input_,
    data_format='json'
)

# Retrieve the results
task_info = client.task.get(task['id'], include_results=True)
while not task_info.get('complete'):
    task_info = client.task.get(task['id'], include_results=True)
    time.sleep(1)
result_info = client.result.list(task=task_info['id'])
results = result_info['data'][0]['result']
```

## Testing locally

If you wish to test the algorithm locally, you can create a Python virtual 
environment, using your favourite method, and do the following:

``` bash
source .venv/bin/activate
pip install -e .
python v6_kmeans_py/example.py
```

The algorithm was developed and tested with Python 3.7.
