# Credit Risk - Batch

This example model is based on the [UCI Taiwan Credit dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). With this dataset, we use a binary classifier for predicting credit default risk from various attributes about payment information and customer demographics. 

## Setup

Create a virtual environment using your favorite environment management tool and install the requirements. As an example,

```commandline
python3 -m venv env
source env/bin/activate

pip3 install arthurai
pip3 install -r requirements.txt
```

Note that the `requirements.txt` file in this directory assumes python versions `3.6-3.8`, as these are currently the only supported versions for the arthur SDK.

## Quickstart

The notebook [notebooks/Quickstart.ipynb](notebooks/Quickstart.ipynb) provides a walkthrough of how to onboard batch models. The interface is largely similar to streaming models, though batches of data can be sent either as DataFrames or parquet files.
