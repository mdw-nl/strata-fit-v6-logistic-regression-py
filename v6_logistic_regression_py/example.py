# -*- coding: utf-8 -*-

""" Sample code to test the federated algorithm with a mock client
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from vantage6.tools.mock_client import ClientMockProtocol


# Start mock client
data_dir = os.path.join(
    os.getcwd(), 'v6_logistic_regression_py', 'local'
)
client = ClientMockProtocol(
    datasets=[
        os.path.join(data_dir, 'data1.csv'),
        os.path.join(data_dir, 'data2.csv')
    ],
    module='v6_logistic_regression_py'
)

# Get mock organisations
organizations = client.get_organizations_in_my_collaboration()
print(organizations)
ids = [organization['id'] for organization in organizations]

# Check master method
master_task = client.create_new_task(
    input_={
        'master': True,
        'method': 'master',
        'kwargs': {
            'org_ids': [0, 1],
            'predictors': ['t', 'n', 'm'],
            'outcome': 'vital_status',
            'classes': ['alive', 'dead'],
            'max_iter': 100,
            'delta': 0.0001
        }
    },
    organization_ids=[0, 1]
)
results = client.get_results(master_task.get('id'))
model = results[0]['model']
iteration = results[0]['iteration']
print(model.coef_, model.intercept_)
print(f'Number of iterations: {iteration}')
