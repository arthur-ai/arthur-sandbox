import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os.path
from os import path
import tempfile
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

    
def _download_arthur_dataset(name, filename, s3_client):
    '''
    Gets a dataset from the public Arthur S3 bucket
    '''

    # makes a folder to store the downloaded data in case it doesnt exist yet
    if not os.path.isdir("arthur_s3_data/"): os.mkdir("arthur_s3_data/")

    if '.zip' in filename:
        with tempfile.TemporaryDirectory() as tempdir:
            target_path = path.join(tempdir, "data.zip")
            try:
                s3_client.Object("s3-bucket-arthur-public", f"sandbox/{name}/data/{filename}").download_file(target_path)
            except Exception as e:
                raise Exception(f'unable to download {filename} from {name} sandbox: {e}')

            try:
                with zipfile.ZipFile(target_path, 'r') as zf:
                    zf.extractall()
                    print(f'downloaded {name} dataset and saved to local arthur_s3_data folder')
            except Exception as e:
                raise Exception(f'unable to unzip dataset {filename}: {e}')
    else:
        try:
            s3_client.Object("s3-bucket-arthur-public", f"sandbox/{name}/data/{filename}").download_file(f"arthur_s3_data/{filename}")
            print(f'downloaded {name} dataset and saved to local data folder')
        except Exception as e:
            raise Exception(f'unable to download {filename} from {name} sandbox: {e}')
    

def _get_arthur_dataset_filetype(name):
    '''
    Gets the file type for a dataset in the public Arthur S3 bucket by its name
    
    name: dataset name
    
    Requirements for this function to work across all sandbox projects:
    - we need a map for this function to use that knows the file type for each dataset
    '''
    file_type=None
    file_types = {'csv' : ['credit_card_default', 'credit_card_default_drifty'], 'zip' : ['fmow']}
    for f_type, dataset_names in file_types.items():
        if name in dataset_names: 
            return f_type
    raise ValueError(f'Could not find the file type for {name}, please ensure {name} is a valid Arthur dataset name.')
    

def _get_arthur_model_filetype(name):
    '''
    Gets the file type for a model in the public Arthur S3 bucket by its name
    
    name: model name
    
    Requirements for this function to work across all sandbox projects:
    - we need a map for this function to use that knows the file type for each model
    '''
    file_type=None
    file_types = {'pkl' : ['fico'], 'h5' : ['yolo_voc']}
    for f_type, model_names in file_types.items():
        if name in model_names: 
            return f_type
    raise ValueError(f'Could not find the file type for {name}, please ensure {name} is a valid Arthur pretrained model name.')
    
    
def _download_arthur_pretrained_model(name, filename, s3_client):
    '''
    Gets a pretrained model from the public Arthur S3 bucket
    '''
    # makes a folder to store the downloaded model in case it doesnt exist yet
    if not os.path.isdir("arthur_s3_models/"): os.mkdir("arthur_s3_models/")

    try:
        s3_client.download_file('s3-bucket-arthur-public', f'sandbox/{name}/models/{filename}', f'arthur_s3_models/{filename}')
        print(f'downloaded {name} model and saved to local arthur_s3_models folder')
    except:
        raise Exception(f'unable to download {filename} from sandbox')

    
def get_arthur_dataset(name):
    '''
    Gets an Arthur dataset.
    
    If the dataset is not alreaedy downlaoded, gets it from the public Arthur S3 bucket.
    
    Requirements for this function to work across all sandbox projects:
    - we need to provide a table of valid dataset names on our documentation
    - we need to format our file names and folder structure consistently in the public bucket
    '''
    file_type = _get_arthur_dataset_filetype(name)
    filename = f"{name}.{file_type}"
    
    # download the data if it does not currently exist in a local folder
    download = not path.exists(f"arthur_s3_data/{filename}")
    
    if download:
        s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        _download_arthur_dataset(name, filename, s3_client)
            
            
def get_arthur_pretrained_model(name):
    '''
    Gets a pretrained Arthur model.
    
    If the model is not alreaedy downlaoded, gets it from the public Arthur S3 bucket.
    
    Requirements for this function to work across all sandbox projects:
    - user provides the name of the model
    - we need to provide a table of valid model names on our documentation
    - we need to format our model names consistently in the public bucket
    - we need a map for this function to use that knows the file type for each model
    '''
    file_type = _get_arthur_model_filetype(name)
    filename = f"{name}.{file_type}"
    
    # download the data if it does not currently exist in a local folder
    download = not path.exists(f"arthur_s3_models/{filename}")
    
    if download:
        s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        _download_arthur_pretrained_model(name, filename, s3_client)

def get_arthur_pretrained_model_predictions(name):
    '''

    '''
    filename = f"{name}_pred_proba.parquet.gzip"

    download = not path.exists(f"arthur_s3_data/{filename}")

    if download:
        s3_client = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        s3_client.Object("s3-bucket-arthur-public", f"sandbox/{name}/data/{filename}").download_file(f"arthur_s3_data/{filename}")
        print(f'downloaded {name} model predictions and saved to local data folder')
    return pd.read_parquet(f"arthur_s3_data/{filename}")

def get_prepared_arthur_dataset(name):
    '''
    Gets a cleaned downloaded dataset
    '''

    # make sure data is downloaded from S3
    get_arthur_dataset(name)

    # get filename based on the user-provided dataset name
    file_type = _get_arthur_dataset_filetype(name)
    filename = f"{name}.{file_type}"

    # prepare each dataset differently depending on features
    if 'credit_card_default' in name:

        # if the data has not already been previously prepared, prepare it from scratch
        # otherwise, load it from a file
        prepare = not path.exists(f"arthur_s3_data/{name}_xtrain.csv")
        prepare = prepare or not path.exists(f"arthur_s3_data/{name}_xtest.csv")
        prepare = prepare or not path.exists(f"arthur_s3_data/{name}_ytrain.csv")
        prepare = prepare or not path.exists(f"arthur_s3_data/{name}_ytest.csv")
        prepare = prepare or not path.exists(f"arthur_s3_data/{name}_sensitive_data.csv")

        if prepare:

            dataset = pd.read_csv(f"arthur_s3_data/{filename}")
            columns = dataset.columns
            
            # separate features by stage
            sensitive_attributes = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
            target = 'default payment next month'
            unique_id = 'ID'
            features = [c for c in columns if c not in sensitive_attributes and c not in [unique_id, target]]

            # prepare model train and test dataframes
            X = dataset[features]
            Y = dataset[[target]]
            # rename the target variable `ground_truth_credit_default`
            Y.rename(columns={target: 'ground_truth_credit_default'}, inplace = True)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=278487)

            # prepare sensitive attribute dataframe
            sensitive_data = dataset[sensitive_attributes]

            # save dataframes to csv
            X_train.to_csv(f'arthur_s3_data/{name}_xtrain.csv')
            X_test.to_csv(f'arthur_s3_data/{name}_xtest.csv')
            Y_train.to_csv(f'arthur_s3_data/{name}_ytrain.csv')
            Y_test.to_csv(f'arthur_s3_data/{name}_ytest.csv')
            sensitive_data.to_csv(f'arthur_s3_data/{name}_sensitive_data.csv')

        # if the data has already been previously prepared, just load it
        else:

            # load dataframes from csv
            X_train = pd.read_csv(f'arthur_s3_data/{name}_xtrain.csv', index_col=0)
            X_test = pd.read_csv(f'arthur_s3_data/{name}_xtest.csv', index_col=0)
            Y_train = pd.read_csv(f'arthur_s3_data/{name}_ytrain.csv', index_col=0)
            Y_test = pd.read_csv(f'arthur_s3_data/{name}_ytest.csv', index_col=0)
            sensitive_data = pd.read_csv(f'arthur_s3_data/{name}_sensitive_data.csv', index_col=0)

    return X_train, X_test, Y_train, Y_test, sensitive_data

        