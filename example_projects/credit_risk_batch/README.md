
# Credit Risk - Batch

This example model is based on the [UCI Taiwan Credit dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). With this dataset, we use a binary classifier for predicting credit default risk from various attributes about payment information and customer demographics. 


### Setup 

Create a virtual environment using your favorite environment management tool and install the requirements. As an example,

```commandline
python3 -m venv env
source env/bin/activate

pip3 install -r requirements.txt
pip3 install arthurai==alpha-1.0.76 --index_url --index-url https://repository.arthur.ai/repository/pypi-virtual/simple
```

#### Onboarding a model and sending inferences
The accompanying [notebook](./notebooks/Quickstart.ipynb) provides a walkthrough of how to onboard batch models. The interface is largely similar to streaming models, though we parquet files to compressing and transferring batches of inferences.
