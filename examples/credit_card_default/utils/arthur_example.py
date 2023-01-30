from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from .communication import get_arthur_dataset, get_arthur_pretrained_model_predictions, get_arthur_example_metadata

local_destination_folder = "arthur_s3_data"
local_destination_model_folder = "arthur_s3_models"


class ArthurExample:
    '''
    Class to build a dataset for an example of using Arthur

    Accesses data from S3 using ArthurSandboxCommunication methods

    Then, saves compiled datasets to CSV files and assigns relevant onboarding dataframes as attributes of this class
    '''

    DEFAULT_SEED = 278487

    def __init__(self, example_name):
        # downloading all relevant data from S3 and save to CSV
        self.example_metadata = get_arthur_example_metadata(example_name)
        file_type = self.example_metadata['dataset_filetype']
        predictions_filetype = self.example_metadata['predictions_filetype']
        get_arthur_dataset(example_name, file_type)
        get_arthur_pretrained_model_predictions(example_name, predictions_filetype)

        # loading the downloaded dataframes
        self.dataset = pd.read_csv(f"{local_destination_folder}/{example_name}.{file_type}", index_col=0)
        self.predictions = pd.read_parquet(f"{local_destination_folder}/{example_name}_pred_proba.{predictions_filetype}")

    @property
    def feature_columns(self) -> List[str]:
        return [c for c in self.dataset.columns
                if c not in self.example_metadata['sensitive_attributes']
                and c not in self.example_metadata['targets']
                and c != self.example_metadata['unique_id']]

    def get_inputs(self, split=True):
        X = self.dataset[self.feature_columns]
        if split:
            return train_test_split(X, random_state=self.DEFAULT_SEED,
                                    test_size=self.example_metadata['data_split_test_size'])
        else:
            return X

    def get_labels(self, split=True):
        targets = self.example_metadata['targets']
        col_rename_map = {t: self.example_metadata['ground_truth_attributes'][i] for i, t in enumerate(targets)}
        y = self.dataset[targets].rename(columns=col_rename_map)
        if split:
            return train_test_split(y, random_state=self.DEFAULT_SEED,
                                    test_size=self.example_metadata['data_split_test_size'])
        else:
            return y

    def get_sensitive_data(self, split=True):
        sensitive_data = self.dataset[self.example_metadata['sensitive_attributes']]
        if split:
            return train_test_split(sensitive_data, random_state=self.DEFAULT_SEED,
                                    test_size=self.example_metadata['data_split_test_size'])
        else:
            return sensitive_data

    def get_predictions(self, split=True):
        if split:
            return train_test_split(self.predictions, random_state=self.DEFAULT_SEED,
                                    test_size=self.example_metadata['data_split_test_size'])
        else:
            return self.predictions


def get_arthur_example(name):
    return ArthurExample(name)
