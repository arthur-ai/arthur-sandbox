import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

ARTHUR_BUCKET = 's3-bucket-arthur-public'
S3_SAGEMAKER_FOLDER = 'sagemaker-demo'

BASE_FOLDER = Path(__file__).parent.parent
MODEL_FOLDER = BASE_FOLDER / "model"
MODEL_METADATA_FILENAME = "model.tar.gz"
MODEL_METADATA_PATH = MODEL_FOLDER / MODEL_METADATA_FILENAME

DATA_FOLDER = BASE_FOLDER / "data"
REFERENCE_DATA_FILENAME = "reference_data.csv"
REFERENCE_DATA_PATH = DATA_FOLDER / REFERENCE_DATA_FILENAME
TEST_DATA_FILENAME = "test_data.csv"
TEST_DATA_PATH = DATA_FOLDER / TEST_DATA_FILENAME


s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def download_model(skip_if_exists=True):
    if not (os.path.exists(MODEL_FOLDER) and os.path.isdir(MODEL_FOLDER)):
        os.mkdir(MODEL_FOLDER)
    
    # if the model.tar.gz file already exists in the model folder, skip download by default
    if (not os.path.isfile(MODEL_METADATA_PATH)) or (not skip_if_exists):
        print("model weights not found, downloading...", end=" ")
        s3_client.download_file(ARTHUR_BUCKET, f"{S3_SAGEMAKER_FOLDER}/{MODEL_METADATA_FILENAME}", str(MODEL_METADATA_PATH))
        print("done!")

        
def download_reference_dataset(skip_if_exists=True):
    if not (os.path.exists(DATA_FOLDER) and os.path.isdir(DATA_FOLDER)):
        os.mkdir(DATA_FOLDER)
    
    # if the reference_data.csv file already exists in the data folder, skip download by default
    if (not os.path.isfile(REFERENCE_DATA_PATH)) or (not skip_if_exists):
        print("reference dataset not found, downloading...", end=" ")
        s3_client.download_file(ARTHUR_BUCKET, f"{S3_SAGEMAKER_FOLDER}/{REFERENCE_DATA_FILENAME}", str(REFERENCE_DATA_PATH))
        print("done!")


def download_test_dataset(skip_if_exists=True):
    if not (os.path.exists(DATA_FOLDER) and os.path.isdir(DATA_FOLDER)):
        os.mkdir(DATA_FOLDER)
    
    # if the test_data.csv file already exists in the data folder, skip download by default
    if (not os.path.isfile(TEST_DATA_PATH)) or (not skip_if_exists):
        print("test dataset not found, downloading...", end=" ")
        s3_client.download_file(ARTHUR_BUCKET, f"{S3_SAGEMAKER_FOLDER}/{TEST_DATA_FILENAME}", str(TEST_DATA_PATH))
        print("done!")
