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
from typing import Dict, List
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data

from v6_logistic_regression_py.helper import coordinate_task
from v6_logistic_regression_py.helper import (
    aggregate,
    export_model,
    initialize_model,
)

MODEL_ATTRIBUTE_KEYS = ["coef_", "intercept_", "classes_"]
MODEL_AGGREGATION_KEYS = ["coef_", "intercept_"]

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
    info('Initializing logistic regression estimator')
    model_initial_attributes = dict(
        classes_  =np.array(classes),
        coef_     =np.zeros((1, len(predictors))),
        intercept_=np.zeros((1,))
    )
    global_model = initialize_model(LogisticRegression, model_initial_attributes)
    model_attributes = export_model(global_model, attribute_keys=MODEL_ATTRIBUTE_KEYS)
    info(model_attributes)

    # The next steps are run until the maximum number of iterations or
    # convergence is reached
    iteration = 0
    loss = None
    loss_diff = 2*delta
    while (iteration < max_iter) and (loss_diff > delta):
        # The input for the partial algorithm
        info(f'######## ITERATION #{iteration} #########')
        input_ = {
            'method': 'logistic_regression_partial',
            'kwargs': {
                'model_attributes': model_attributes,
                'predictors': predictors,
                'outcome': outcome
            }
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)
        info(f'Results before aggregation: {results}')

        # # Reassign model parameters
        # global_model = update_model(global_model, model_attributes=results['model_attributes'])

        # # Average model weights with weighted average
        # info(f'Run global averaging for model weights: {results}')
        # coefficients = np.zeros((1, len(predictors)))
        # for i in range(coefficients.shape[1]):
        #     coefficients[0, i] = np.sum([
        #         result['model'].coef_[0, i]*result['size']
        #         for result in results
        #     ]) / np.sum([
        #         result['size'] for result in results
        #     ])
        # intercept = np.sum([
        #     result['model'].intercept_*result['size'] for result in results
        # ]) / np.sum([
        #     result['size'] for result in results
        # ])
        # intercept = np.array([intercept])

        # Re-define the global parameters
        # parameters = (coefficients, intercept)
        # model = set_model_params(model, parameters)

        # Aggregate the results
        info("Aggregating partial modeling results")
        global_model = aggregate(global_model, results=results, aggregation_keys=MODEL_AGGREGATION_KEYS)
        info("Exporting global model")
        global_model_attributes = export_model(model=global_model, attribute_keys=MODEL_ATTRIBUTE_KEYS)

        # The input for the partial algorithm that computes the loss
        info('Computing local losses')
        input_ = {
            'method': 'compute_loss_partial',
            'kwargs': {
                'model_attributes': global_model_attributes,
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

    return {
        'model_attributes': global_model_attributes,
        'loss': loss,
        'iteration': iteration
    }


@data(1)
def logistic_regression_partial(
        df: pd.DataFrame,
        model_attributes: Dict[str, List[float]],
        predictors,
        outcome
) -> dict:
    """ Partial method for federated logistic regression

    Parameters
    ----------
    df
        DataFrame with input data
    model_attributes
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

    # Create local LogisticRegression estimator object
    model_kwargs = dict(
        max_iter=1,       # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    model = initialize_model(LogisticRegression, model_attributes=model_attributes, **model_kwargs)
    
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
        info('Training round finished')
    
    model_attributes = export_model(model, attribute_keys=MODEL_ATTRIBUTE_KEYS)
    info(f'MODEL ATTRIBUTES: {model_attributes}')

    return {
        'model_attributes': model_attributes,
        'size': X.shape[0]
    }


@data(1)
def compute_loss_partial(
        df: pd.DataFrame,
        model_attributes: Dict[str, list],
        predictors,
        outcome
) -> dict:
    """ Partial method for calculation of loss

    Parameters
    ----------
    df
        DataFrame with input data
    model_attributes
        Serializable model parameters in Dict[str, list] format
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

    # Initialize local model instance
    model = initialize_model(LogisticRegression, model_attributes)

    # Compute loss
    loss = log_loss(y, model.predict_proba(X))

    return {
        'loss': loss,
        'size': X.shape[0]
    }


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

    # Initialize LogisticRegression estimator
    model_attributes=dict(
            intercept_ = np.array(parameters[0]),
            coef_ = np.array(parameters[1]),
            classes_ = np.array(classes)
            )
    model = initialize_model(LogisticRegression, model_attributes)

    # Compute model accuracy
    score = model.score(X, y)

    # Compute confusion matrix
    confusion_matrix_ = confusion_matrix(
        y, model.predict(X), labels=model.classes_
    ).tolist()

    return {
        'score': score,
        'confusion_matrix': confusion_matrix_
    }