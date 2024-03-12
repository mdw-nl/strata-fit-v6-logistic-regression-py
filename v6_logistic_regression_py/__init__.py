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
from sklearn.metrics import confusion_matrix
from typing import List
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data

from v6_logistic_regression_py.helper import coordinate_task
from v6_logistic_regression_py.helper import set_initial_params
from v6_logistic_regression_py.helper import set_model_params
from v6_logistic_regression_py.helper import get_model_parameters
from v6_logistic_regression_py.helper import init_model, export_model

MODEL_ATTRIBUTE_KEYS = ["coef_", "intercept_", ]

@algorithm_client
def master(
        client: AlgorithmClient,
        predictors: List[str],
        outcome: str,
        classes: list,
        max_iter: int = 15,
        delta: float = 0.01,
        org_ids: List[int] = None
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
    info('Collecting the identification of the participating organizations')
    organizations = client.organization.list()
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

@data(1)
def logistic_regression_partial(
        df: pd.DataFrame,
        parameters,
        predictors,
        outcome
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
    
    model_dict = export_model(model, attribute_keys=[])

    # Results
    results = {
        'model': model,
        'size': X.shape[0]
    }

    return results


@data(1)
def compute_loss_partial(
        df: pd.DataFrame,
        model,
        predictors,
        outcome
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


@data(1)
def run_validation(
        df: pd.DataFrame,
        parameters: list,
        classes: list,
        predictors: list,
        outcome: str
) -> dict:
    """ Method for running model validation

    Parameters
    ----------
    df
        DataFrame with input data
    parameters
        List with coefficients
    classes
        Classes to be predicted
    predictors
        List with columns to be used as predictors
    outcome
        Column to be used as outcome

    Returns
    -------
    performance
        Dictionary with performance metrics
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Logistic regression model
    model = init_model(
        LogisticRegression,
        model_attributes=dict(
            intercept_ = np.array(parameters[0]),
            coef_ = np.array(parameters[1]),
            classes_ = np.array(classes)
            )
        )
    # model = LogisticRegression()
    # model.coef_ = np.array(parameters[1])
    # model.intercept_ = np.array(parameters[0])
    # model.classes_ = np.array(classes)

    # Compute model accuracy
    score = model.score(X, y)

    # Confusion matrix
    cm = confusion_matrix(
        y, model.predict(X), labels=model.classes_
    )

    # Results
    results = {
        'score': score,
        'confusion_matrix': cm
    }

    return results
