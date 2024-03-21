""" Federated logistic regression algorithm adapted from Flower's federated scikit-learn example:
https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/
"""
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from typing import Any, Dict, List
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data

from v6_logistic_regression_py.helper import (
    aggregate,
    coordinate_task,
    export_model,
    initialize_model
)

MODEL_ATTRIBUTE_KEYS = ["coef_", "intercept_", "classes_"]
MODEL_AGGREGATION_KEYS = ["coef_", "intercept_"]


@algorithm_client
def master(
    client: AlgorithmClient,
    predictors: List[str],
    outcome: str,
    classes: List[str],
    max_iter: int = 15,
    delta: float = 0.01,
    n_local_iterations: int = 1,
    org_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Orchestrates federated logistic regression training across nodes.

    Parameters
    ----------
    client : AlgorithmClient
        Vantage6 user or mock client.
    predictors : List[str]
        Columns to be used as predictors.
    outcome : str
        Column to be used as target variable.
    classes : List[str]
        List of class labels.
    max_iter : int, optional
        Maximum number of iterations for convergence.
    delta : float, optional
        Convergence threshold based on loss difference.
    org_ids : List[int], optional
        Specific organization IDs to involve in computation.

    Returns
    -------
    Dict[str, any]
        Aggregated model attributes, last loss value, and number of iterations performed.
    """

    # Identifying data nodes participating in the federated learning process.
    info("Identifying participating organizations.")
    organizations = client.organization.list()
    ids = [org['id'] for org in organizations if not org_ids or org['id'] in org_ids]

    # Initializing the global logistic regression model with zero weights.
    info("Initializing global logistic regression model.")
    model_attrs = {
        'classes_': np.array(classes),
        'coef_': np.zeros((1, len(predictors))),
        'intercept_': np.zeros(1)
    }
    global_model = initialize_model(LogisticRegression, model_attrs)

    # Iteratively updating the global model based on local updates until convergence.
    iteration, loss, loss_diff = 0, None, 2 * delta
    while iteration < max_iter and loss_diff > delta:
        info(f"Starting iteration #{iteration + 1}.")

        # Sending model training tasks to nodes and collecting updates.
        model_attrs = export_model(global_model, MODEL_ATTRIBUTE_KEYS)
        input_ = {
            'method': 'logistic_regression_partial',
            'kwargs': {'model_attributes': model_attrs, 'predictors': predictors, 'outcome': outcome, "n_local_iterations": n_local_iterations}
            }
        partial_results = coordinate_task(client, input_, ids)

        # Aggregating updates into the global model and assessing convergence.
        global_model = aggregate(global_model, partial_results, MODEL_AGGREGATION_KEYS)
        new_loss = compute_global_loss(client, global_model, predictors, outcome, ids)
        
        loss_diff = abs(loss - new_loss) if loss is not None else 2 * delta
        loss = new_loss
        info(f"Iteration #{iteration + 1} completed. Loss difference: {loss_diff}.")

        iteration += 1

    info("Federated training completed.")

    return {
        'model_attributes': export_model(global_model, MODEL_ATTRIBUTE_KEYS),
        'loss': loss,
        'iteration': iteration
    }

def compute_global_loss(client, model, predictors, outcome, ids):
    """
    Helper function to compute global loss, abstracting detailed logging.
    """
    model_attributes = export_model(model, MODEL_ATTRIBUTE_KEYS)
    input_ = {
        'method': 'compute_loss_partial',
        'kwargs': {'model_attributes': model_attributes, 'predictors': predictors, 'outcome': outcome}
        }
    results = coordinate_task(client, input_, ids)
    aggregated_sample_size = np.sum([res['size'] for res in results])
    aggregated_loss = np.sum([res['loss'] * res['size'] for res in results])
    new_loss = aggregated_loss / aggregated_sample_size
    return new_loss


@data(1)
def logistic_regression_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, List[float]], 
    predictors: List[str], 
    outcome: str,
    n_local_iterations: int = 1,
) -> Dict[str, any]:
    """
    Fits logistic regression model on local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame.
    model_attributes : Dict[str, List[float]]
        Logistic regression model attributes (weights, intercepts).
    predictors : List[str]
        List of predictor variable names.
    outcome : str
        Outcome variable name.

    Returns
    -------
    Dict[str, any]
        Attributes of locally trained logistic regression model and local dataset size.
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Create local LogisticRegression estimator object
    model_kwargs = dict(
        max_iter=n_local_iterations,       # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    model = initialize_model(LogisticRegression, model_attributes=model_attributes, **model_kwargs)
    
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
        info('Training round finished')
    
    model_attributes = export_model(model, attribute_keys=MODEL_ATTRIBUTE_KEYS)

    return {
        'model_attributes': model_attributes,
        'size': X.shape[0]
    }


@data(1)
def compute_loss_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, list], 
    predictors: List[str], 
    outcome: str
) -> Dict[str, Any]:
    """
    Computes logistic regression model loss on local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame.
    model_attributes : Dict[str, list]
        Attributes of the logistic regression model.
    predictors : List[str]
        Predictor variables.
    outcome : str
        Outcome variable.

    Returns
    -------
    Dict[str, Any]
        Local loss and dataset size.
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
    parameters: List[np.ndarray], 
    classes: List[str], 
    predictors: List[str], 
    outcome: str
) -> Dict[str, Any]:
    """
    Validates logistic regression model on local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame for validation.
    parameters : List[np.ndarray]
        Model parameters for validation.
    classes : List[str]
        List of class labels.
    predictors : List[str]
        Predictor variables for validation.
    outcome : str
        Outcome variable for validation.

    Returns
    -------
    Dict[str, Any]
        Performance metrics including model accuracy and confusion matrix.
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