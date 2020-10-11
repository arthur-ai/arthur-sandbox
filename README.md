[![image](https://static1.squarespace.com/static/5c2161df4cde7a0c1a70b37c/t/5c227d398a922d984e29518b/1587145309881/?format=300w)](https://arthur.ai)
# Getting Started with Arthur

Thanks for checking out the Arthur Examples. In this repo, you'll find sample projects of ML models and how to integrate them with Arthur. You'll also find helpful walkthroughs and guides of how to get the most of our our python client library.

## Installation

Install the SDK via pip: an Arthur engineer will provide you with credentials to install the latest version. For example:

`pip install arthurai --index-url https://repository.arthur.ai/repository/pypi-virtual/simple`. 

#### Troubleshooting

If you see the following error message:

```
ERROR: Could not install packages due to an EnvironmentError: HTTPSConnectionPool(host='repository.arthur.ai', port=443): Max retries exceeded with url: <url> (Caused by ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
```

You may need to add the `--no-cache` flag when pip installing.

#### Using Pipenv
You can also install using the included Pipfile:
1. Export the username of the provided credentials as: `export NEXUS_REPOSITORY_USERNAME=USERNAME`
1. Export the password of the provided credentials as: `export NEXUS_REPOSITORY_PASSWORD=PASSWORD`
1. Run the following command to handle the issue noted above: `export PIP_NO_CACHE_DIR=false`
1. `pipenv install --skip-lock`
1. To enable virtualenv in your shell do `pipenv shell`

#### Python Versions and Package Management

The `requirements.txt` files in each of the example project directories (`boston_housing_spark_model`, `credit_risk`, `credit_risk_batch`) support python versions `3.6-3.8`.  If using your own model, or if using a python3 version `3.5` or earlier, you will need to update the `requirements.txt` files with compatible package versions. Specifically, ensure that the `requirements.txt` file has the exact package versions you are using in your local environment.

## Projects
If it's your first time onboarding your model to Arthur, check out some of the [example ML projects](./example_projects) to help inspire you. 

#### [Credit Model](./example_projects/credit_risk/README.md)
 * [Quick Onboarding](example_projects/credit_risk/onboard_quick.py)
 * [Full Onboarding](example_projects/credit_risk/onboard_full.py): includes setting up explainability, preparing for datadrift monitoring, and registering sensitive attriutes for bias monitoring.


## Guides
* [Pulling Metrics](./SDK_examples/sdk_retrieve_metrics_and_data.ipynb): includes examples of fetching metrics from the platform for further analysis and visualization.
* [Quickstart](./example_projects/credit_risk/notebooks/Quickstart.ipynb): Demo of onboarding and sending inferences with a credit risk model. 
* [Batch Model Quickstart](./example_projects/credit_risk_batch/notebooks/Quickstart.ipynb): Demo of onboarding and sending inferences with a batch model. 


## Docs
For full detail about the SDK and API, refer to the [Arthur documentation](docs.arthur.ai).
