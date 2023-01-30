import boto3
from botocore import UNSIGNED
from botocore.client import Config
import codecs
import csv
import os.path
from os import path
import tempfile
import zipfile
import pandas as pd
import yaml

local_destination_folder = "arthur_s3_data"
local_destination_model_folder = "arthur_s3_models"


def get_arthur_dataset(name, file_type):
    '''
    Gets an Arthur dataset.
    '''
    if not os.path.isdir(local_destination_folder + "/"): os.mkdir(local_destination_folder + "/")

    filename = f"{name}.{file_type}"

    # download the data if it does not currently exist in a local folder
    download = not path.exists(f"{local_destination_folder}/{name}/{filename}")

    if download:
        # s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        response = s3_client.get_object(Bucket="s3-bucket-arthur-public", Key=f"sandbox/{name}/data/{filename}")
        if file_type == "csv":
            csvreader = csv.DictReader(codecs.getreader("utf-8")(response["Body"]))
            newcsv = pd.DataFrame([row for row in csvreader])
            newcsv.to_csv(f"{local_destination_folder}/{filename}")
        elif file_type == "zip":
            with tempfile.TemporaryDirectory() as tempdir:
                target_path = path.join(tempdir, "data.zip")

                s3_client.Object("s3-bucket-arthur-public", f"sandbox/{name}/data/{filename}").download_file(
                    target_path)
                with zipfile.ZipFile(target_path, 'r') as zf:
                        zf.extractall()


def get_arthur_pretrained_model(name, file_type):
    '''
    Gets a pretrained Arthur model.
    '''
    if not os.path.isdir(local_destination_model_folder + "/"): os.mkdir(local_destination_model_folder + "/")

    filename = f"{name}.{file_type}"

    # download the data if it does not currently exist in a local folder
    download = not path.exists(f"{local_destination_model_folder}/{filename}")

    if download:
        s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

        s3_client.download_file('s3-bucket-arthur-public', f'sandbox/{name}/models/{filename}',
                                f'{local_destination_model_folder}/{filename}')


def get_arthur_pretrained_model_predictions(name, file_type):
    '''
    Gets a set of predictions from a pretrained Arthur model
    '''
    filename = f"{name}_pred_proba.{file_type}"

    download = not path.exists(f"{local_destination_folder}/{filename}")

    if download:
        s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        print(filename)
        s3_client.Object("s3-bucket-arthur-public", f"sandbox/{name}/predictions/{filename}").download_file(
            f"{local_destination_folder}/{filename}")
        print(f'downloaded {name} model predictions and saved to local {local_destination_folder} folder')



def get_arthur_example_metadata(name):
    '''
    Gets metadata for an Arthur example from a yaml file in the sandbox folder for this example
    '''

    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3_client.get_object(Bucket="s3-bucket-arthur-public", Key=f"sandbox/{name}/{name}_metadata.yaml")
    example_metadata = yaml.safe_load(response["Body"])
    assert example_metadata['name'] == name
    return example_metadata
