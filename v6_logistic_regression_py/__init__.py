# -*- coding: utf-8 -*-

""" Federated algorithm for logistic regression
Adapted from:
https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/
"""
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from vantage6.tools.util import info

from v6_logistic_regression_py.helper import coordinate_task
from v6_logistic_regression_py.helper import set_initial_params
from v6_logistic_regression_py.helper import set_model_params
from v6_logistic_regression_py.helper import get_model_parameters


def master(
        client, data: pd.DataFrame, predictors: list, outcome: str,
        classes: list, max_iter: int = 15, delta: float = 0.01,
        org_ids: list = None
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
    classes
        List with classes to be predicted
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
    model = set_initial_params(model, len(predictors), classes)
    parameters = get_model_parameters(model)

    # The next steps are run until the maximum number of iterations or
    # convergence is reached
    iteration = 0
    loss = None
    loss_diff = 2*delta
    while (iteration < max_iter) and (loss_diff > delta):
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

        # Re-define the global parameters
        parameters = (coefficients, intercept)
        model = set_model_params(model, parameters)

        # The input for the partial algorithm that computes the loss
        info('Computing local losses')
        input_ = {
            'method': 'compute_loss_partial',
            'kwargs': {
                'model': model,
                'predictors': predictors,
                'outcome': outcome
            }
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)
        info(f'Results: {results}')

        # Aggregating local losses into a global loss
        info('Run global averaging for losses')
        new_loss = np.sum([
            result['loss']*result['size'] for result in results
        ]) / np.sum([
            result['size'] for result in results
        ])

        # Check convergence: we assume convergence when the difference in the
        # global loss between iterations gets below a certain threshold,
        # the difference is set to a value greater than delta in iteration zero
        info('Checking convergence')
        loss_diff = np.abs(loss - new_loss) if iteration != 0 else 2*delta
        loss = new_loss
        info(f'Difference is loss = {loss_diff}')

        # Update iterations counter
        iteration += 1

    # Final result
    result = {
        'model': model,
        'loss': loss,
        'iteration': iteration
    }

    return result


def RPC_logistic_regression_partial(
        df: pd.DataFrame, parameters, predictors, outcome
) -> dict:
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
        Dictionary with local model
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Create LogisticRegression Model
    model = LogisticRegression(
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

    # Results
    results = {
        'model': model,
        'size': X.shape[0]
    }

    return results


def RPC_compute_loss_partial(
        df: pd.DataFrame, model, predictors, outcome
) -> dict:
    """ Partial method for calculation of loss

    Parameters
    ----------
    df
        DataFrame with input data
    model
        Logistic regression model object
    predictors
        List with columns to be used as predictors
    outcome
        Column to be used as outcome

    Returns
    -------
    loss
        Dictionary with local loss
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Compute loss
    loss = log_loss(y, model.predict_proba(X))

    # Results
    results = {
        'loss': loss,
        'size': X.shape[0]
    }

    return results
