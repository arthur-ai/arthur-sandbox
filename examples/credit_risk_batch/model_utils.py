from sklearn.model_selection import train_test_split
from typing import Union, Tuple
import pandas as pd
import numpy as np


def load_datasets(training_filepath: str,
                  pipeline_transformations=(lambda x: x)) \
        -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load training and test data for the credit risk dataset.
    :param training_filepath: filepath to training data csv
    :param: pipeline_transformations: a function closure that compute any necessary
        transformations of raw input data. Defaults to no transformations.
    :return: Tuple(TrainingData, TestData)"""
    credit_df = pd.read_csv(training_filepath)
    X = pipeline_transformations(credit_df.iloc[:, 1:-1])
    Y = credit_df["default payment next month"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return (X_train, Y_train), (X_test, Y_test)

def transformations(input: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """
    Pipeline transformations for the credit model. In this case, transformations
    only include taking the logarithm of the LIMIT_BAL feature.
    :param input: filepath to training data csv
    :return: DataFrame or Ndarray
    """
    if isinstance(input, pd.DataFrame):
        x = input.copy()
        x["LIMIT_BAL"] = np.log(input.LIMIT_BAL)
    elif isinstance(input, np.ndarray):
        x = input.copy()
        x[:, 0] = np.log(input[:, 0])
    elif isinstance(input, list):
        x = np.array(input)
        x[:, 0] = np.log(x[:, 0])
    return x