# -*- coding: utf-8 -*-

""" Helper functions for running logistic regression
"""
import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from typing import Any, Dict, List, Tuple, Type, Union

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]


def coordinate_task(client: AlgorithmClient, input: dict, ids: list) -> list:
    """ Coordinate tasks to be sent to data nodes, which includes dispatching
    the task, waiting for results to return and collect completed results

    Parameters
    ----------
    client
        Vantage6 user or mock client
    input
        Input parameters for the task, such as the method and its arguments
    ids
        List with organisation ids that will receive the task

    Returns
    -------
    results
        Collected partial results from all the nodes
    """

    # Create a new task for the desired organizations
    info('Dispatching node tasks')
    task = client.task.create(
        input_=input,
        organizations=ids
    )

    # Wait for nodes to return results
    info('Waiting for results')
    task_id = task.get('id')
    task = client.get_task(task_id)
    while not task.get('complete'):
        task = client.get_task(task_id)
        info('Waiting for results')
        time.sleep(1)

    # Collecting results
    info('Obtaining results')
    results = client.get_results(task_id=task.get('id'))

    return results


def set_initial_params(model: LogisticRegression, ncoef, classes):
    """Sets initial parameters as zeros"""
    model.classes_ = np.array(classes)
    model.coef_ = np.zeros((1, ncoef))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))
    return model


def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def init_model(
    model_class: Type[BaseEstimator],
    model_attributes: Dict[str, Any],
    *model_init_args: Any, **model_init_kwargs: Any
) -> BaseEstimator:
    """
    Initializes an instance of the provided model class with corresponding model attributes.

    Parameters
    ----------
    model_class : Type[BaseEstimator]
        Class of the model to be initialized. 
    model_attributes : Dict[str, Any]
        Dictionary mapping attribute names to their corresponding values.
    model_init_args : Any
        Positional arguments for model class initialization.
    model_init_kwargs : Any
        Keyword arguments for model class initialization.

    Returns
    -------
    model : BaseEstimator
        The initialized model object.
    """
    model = model_class(*model_init_args, **model_init_kwargs)
    for key, value in model_attributes.items():
        setattr(model, key, value)
    return model


def export_model(model: BaseEstimator, attribute_keys: List[str]) -> Dict[str, Any]:
    """
    Exports model attributes given a model and list of attributes.

    Parameters
    ----------
    model : BaseEstimator
        An instance of the Scikit-Learn's BaseEstimator (such as LogisticRegression, SVC, etc.).
    attribute_keys : List[str]
        List of attribute names that are to be extracted from the model.

    Returns
    -------
    attributes : Dict[str, Any]
        A Dictionary mapping attribute keys to their corresponding values extracted from the model.
    """
    attributes = {key: getattr(model, key) for key in attribute_keys}
    return attributes