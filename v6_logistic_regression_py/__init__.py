# -*- coding: utf-8 -*-

""" Federated algorithm for logistic regression
Adapted from:
https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/
"""
import re
import warnings

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from vantage6.tools.util import info
from v6_logistic_regression_py.helper import coordinate_task
from v6_logistic_regression_py.helper import set_initial_params
from v6_logistic_regression_py.helper import set_model_params
from v6_logistic_regression_py.helper import get_model_parameters


def master(
        client, data: pd.DataFrame, predictors: list, outcome: str,
        max_iter: int = 15, delta: float = 0.01, org_ids: list = None
) -> dict:
    """ Master algorithm that coordinates the tasks and performs averaging

    Parameters
    ----------
    client
        Vantage6 user or mock client
    data
        DataFrame with the input data
    predictors
        List with columns to be used as predictors
    outcome
        Column to be used as outcome
    max_iter
        Maximum number of iterations to perform
    delta
        Threshold for difference between losses to consider convergence
    org_ids
        List with organisation ids to be used

    Returns
    -------
    results
        Dictionary with the final averaged result
    """

    # Get all organization ids that are within the collaboration or
    # use the provided ones
    info('Collecting participating organizations')
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get('id') for organization in organizations
           if not org_ids or organization.get('id') in org_ids]

    # Initialise the weights for the logistic regression
    info('Initializing logistic regression weights')
    model = LogisticRegression()
    model = set_initial_params(model, len(predictors))
    parameters = get_model_parameters(model)

    # TODO: check convergence by looking at the losses
    # The next steps are run until the maximum number of iterations is reached
    iteration = 0
    while iteration < max_iter:
        # The input for the partial algorithm
        info('Defining input parameters')
        input_ = {
            'method': 'logistic_regression_partial',
            'kwargs': {
                'parameters': parameters,
                'predictors': predictors,
                'outcome': outcome
            }
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)
        info(f'Results: {results}')

        # Average model weights with weighted average
        info('Run global averaging for model weights')
        coefficients = np.zeros((1, len(predictors)))
        for i in range(coefficients.shape[1]):
            coefficients[0, i] = np.sum([
                result['model'].coef_[0, i]*result['size']
                for result in results
            ]) / np.sum([
                result['size'] for result in results
            ])
        intercept = np.sum([
            result['model'].intercept_*result['size'] for result in results
        ]) / np.sum([
            result['size'] for result in results
        ])
        intercept = np.array([intercept])

        # # TODO: how to average losses and accuracy?
        # # Average loss and accuracy with weighted average
        # loss = np.sum([
        #     result['loss']*result['size'] for result in results
        # ]) / np.sum([
        #     result['size'] for result in results
        # ])
        # accuracy = np.sum([
        #     result['accuracy']*result['size'] for result in results
        # ]) / np.sum([
        #     result['size'] for result in results
        # ])

        # Re-define the global parameters and update iterations counter
        parameters = (coefficients, intercept)
        model = set_model_params(model, parameters)
        iteration += 1

    return {
        'model': model
        # 'loss': loss,
        # 'accuracy': accuracy
    }


def RPC_logistic_regression_partial(
        df: pd.DataFrame, parameters, predictors, outcome
) -> list:
    """ Partial method for federated logistic regression

    Parameters
    ----------
    df
        DataFrame with input data
    parameters
        Model weigths of logistic regression
    predictors
        List with columns to be used as predictors
    outcome
        Column to be used as outcome

    Returns
    -------
    results
        Dictionary with local model, loss and accuracy
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty='l2',
        max_iter=1,       # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Fitting local model
    model = set_model_params(model, parameters)
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
        info('Training round finished')

    # # Evaluate model
    # loss = log_loss(y_test, model.predict_proba(X_test))
    # accuracy = model.score(X_test, y_test)

    # Results
    results = {
        'model': model,
        # 'loss': loss,
        # 'accuracy': accuracy,
        'size': X.shape[0]
    }

    return results
