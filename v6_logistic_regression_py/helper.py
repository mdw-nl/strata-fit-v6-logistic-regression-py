import numpy as np
from sklearn.base import BaseEstimator
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn
from typing import Any, Dict, List, Type, Union


def coordinate_task(client: AlgorithmClient, input_: Dict[str, Any], ids: List[int]) -> List[Dict[str, Any]]:
    """
    Coordinate tasks to be sent to data nodes.

    Parameters
    ----------
    client : AlgorithmClient
        Vantage6 user or mock client.
    input_ : Dict[str, Any]
        Input parameters for the task, such as the method and its arguments.
    ids : List[int]
        List with organisation ids that will receive the task.

    Returns
    -------
    List[Dict[str, Any]]
        Collected partial results from all the nodes.
    """
    info('Dispatching node tasks')
    task = client.task.create(input_=input_, organizations=ids)

    info('Waiting for results')
    results = client.wait_for_results(task_id=task.get("id"), interval=1)
    info(f'Results obtained for {input_["method"]}!')

    return results


def aggregate(global_model: BaseEstimator, results: List[Dict[str, Any]], aggregation_keys: List[str]) -> BaseEstimator:
    """
    Aggregate local results into a global model by weighted average of parameters.

    Parameters
    ----------
    global_model : BaseEstimator
        Global model instance to be updated.
    results : List[Dict[str, Any]]
        List of local results, each containing model attributes and the data size.
    aggregation_keys : List[str]
        Keys whose values are to be aggregated.

    Returns
    -------
    BaseEstimator
        Updated global model with averaged parameters.
    """
    total_data_size = sum(result['size'] for result in results)
    summed_attributes = {key: np.zeros_like(results[0]['model_attributes'][key]) for key in aggregation_keys}

    for result in results:
        for key, value in result['model_attributes'].items():
            if key in aggregation_keys:
                summed_attributes[key] += np.array(value) * result['size']

    aggregated_attributes = {key: value / total_data_size for key, value in summed_attributes.items()}
    return update_model(global_model, aggregated_attributes)


def initialize_model(model_class: Type[BaseEstimator], model_attributes: Dict[str, Any], *model_init_args: Any, **model_init_kwargs: Any) -> BaseEstimator:
    """
    Initializes an instance of the provided model class with corresponding model attributes.

    Parameters
    ----------
    model_class : Type[BaseEstimator]
        Class of the model to be initialized.
    model_attributes : Dict[str, Any]
        Attribute names and their corresponding values.
    model_init_args : Any
        Positional arguments for model class initialization.
    model_init_kwargs : Any
        Keyword arguments for model class initialization.

    Returns
    -------
    BaseEstimator
        The initialized model object.
    """
    model = model_class(*model_init_args, **model_init_kwargs)
    return update_model(model, model_attributes)


def export_model(model: BaseEstimator, attribute_keys: List[str]) -> Dict[str, Any]:
    """
    Exports model attributes given a model and list of attribute keys.

    Parameters
    ----------
    model : BaseEstimator
        An instance of Scikit-Learn's BaseEstimator.
    attribute_keys : List[str]
        Attribute names to be extracted from the model.

    Returns
    -------
    Dict[str, Any]
        Dictionary of attribute keys and their values.
    """
    return {key: to_json_serializable(getattr(model, key)) for key in attribute_keys}


def update_model(model: BaseEstimator, model_attributes: Dict[str, Any]) -> BaseEstimator:
    """
    Updates the model's attributes, attempting to convert lists to numpy arrays.

    Parameters
    ----------
    model : BaseEstimator
        The model to update.
    model_attributes : Dict[str, Any]
        Attributes to update in the model.

    Returns
    -------
    BaseEstimator
        The updated model.
    """
    for key, value in model_attributes.items():
        if isinstance(value, list):
            try:
                value = np.array(value)
            except ValueError:
                warn(f"Conversion failed for {key}, remains a list.")
        setattr(model, key, value)
    return model


def to_json_serializable(item: Union[np.ndarray, dict, Any]) -> Union[list, dict, Any]:
    """
    Converts an item to JSON-serializable format. Numpy arrays are converted to lists.

    Parameters
    ----------
    item : Union[np.ndarray, dict, Any]
        The item to convert.

    Returns
    -------
    Union[list, dict, Any]
        The JSON-serializable representation of the item.
    """
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, dict):
        return {key: to_json_serializable(value) for key, value in item.items()}
    return item

def trash_outcomes(
        df,
        outcome,
        survival_column="Survival.time",
        event_column="deadstatus.event",
        threshold=730):
    df[outcome] = (df[survival_column] <= threshold).astype(int)
    return df 