# -*- coding: utf-8 -*-

""" Helper functions for running logistic regression
"""
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn
from typing import Any, Dict, List, Tuple, Type, Union

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]


def coordinate_task(client: AlgorithmClient, input_: dict, ids: List[int]) -> List[Dict[str, Any]]:
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
        input_=input_,
        organizations=ids
    )

    # Wait for nodes to return results
    info('Waiting for results')
    results = client.wait_for_results(task_id=task.get("id"), interval=1)
    info(f'Results obtained for {input_["method"]}!')

    return results


def aggregate(
    global_model: BaseEstimator,
    results: List[Dict[str, Any]],
    aggregation_keys: List[str]
) -> BaseEstimator:
    """
    Aggregate local results into a global model by weighted average of parameters.

    Parameters
    ----------
    global_model : BaseEstimator
        Global model instance to be updated.
    results: List[Dict[str, Any]]
        List of local results, each a dictionary containing model attributes and the data size.
    aggregation_keys: List[str]
        List of keys, which values have to be aggregated. Other keys' values will be taken from the first site
        
    Returns
    -------
    BaseEstimator
        Updated global model with averaged parameters.
    """
    # Global sample size
    total_data_size = sum(result['size'] for result in results)

    # Initialize dictionary to hold summed attributes using numpy arrays
    summed_attributes = {
        key: np.zeros_like(results[0]['model_attributes'][key])
        for key in results[0]['model_attributes']
        if key in aggregation_keys}
    
    for result in results:
        for key, value in result['model_attributes'].items():
            if key not in aggregation_keys: continue
            # Ensure value is a numpy array for vectorized operations
            current_value = np.array(value)
            summed_attributes[key] += current_value * result['size']  # Weight by size
    info(summed_attributes)

    # Calculate the weighted average of attributes
    aggregated_attributes = {key: value / total_data_size for key, value in summed_attributes.items()}

    # Update the global model with averaged attributes
    global_model = update_model(global_model, aggregated_attributes)
    
    return global_model


def initialize_model(
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
    model = update_model(model, model_attributes)
    return model


def export_model(model: BaseEstimator, attribute_keys: List[str]) -> Dict[str, Any]:
    """
    Exports model attributes given a model and list of attribute keys.

    WARNING: nested dictionaries with numpy values were NOT TESTED

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
    attributes = {key: to_json_serializable(getattr(model, key)) for key in attribute_keys}
    return attributes


def update_model(
    model: BaseEstimator,
    model_attributes: Dict[str, Any]
) -> BaseEstimator:
    """Updates the model's attributes, converting lists to numpy arrays where possible."""
    for key, value in model_attributes.items():
        try:
            # Convert lists to numpy arrays if all elements are numeric
            if isinstance(value, list):
                value = np.array(value)
        except ValueError:
            warn(f"Could not convert {key} attribute to a numpy array.")
        setattr(model, key, value)
    return model


def to_json_serializable(item: Union[np.ndarray, dict, Any]) -> Union[list, dict, Any]:
    """Convert an item to a format that is serializable to JSON."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, dict):
        return {key: to_json_serializable(value) for key, value in item.items()}
    return item