[![image](https://arthurai-assets.s3.amazonaws.com/logo-black-transparent.png?format=300w)](https://arthur.ai)
# Getting Started with Arthur

In this repo, you'll find sample projects of integrating machine learning models with Arthur, with tips on how to get the most out of the Arthur Python SDK.

| Example             | Input Type  | Output Type      | Inference Type | Data Storage     | Model Storage   | Arthur features                                                                                               | Frameworks      |
| ------------------- | ----------- | ---------------- | -------------- | ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------- | --------------- |
| Credit Risk         | Tabular     | Binary           | Batch & stream | Local            | Local           | Performance, Drift, Visualization, Alerts, Bias, Explainability, Anomaly Detection, Hotspots, Bias Mitigation | Scikit-Learn    |
| MEPS Healthcare     | Tabular     | Binary           | Stream         | Local            | Local           | Explainability | Scikit-Learn, NLTK |
| Boston Housing      | Tabular     | Regression       | Batch          | Local            | Local           | Explainability | Spark ML |
| Cancer Detection    | Image       | Binary           | Stream         | Arthur S3 Bucket | Local           | Explainability | PyTorch |
| Mars Rover          | Image       | Object Detection | Stream         | Nasa API and Arthur S3 Bucket        | SageMaker           | Explainability | Scikit-Learn, OpenCV, PyTorch |
| Object Detection    | Image       | Object Detection | Stream         | Arthur S3 Bucket | Local           | Explainability | OpenCV, TensorFlow |
| Satellite Images    | Image       | Binary           | Batch          | Arthur S3 Bucket | Local           | Explainability | PyTorch |
| Medical Transcripts | NLP         | Multiclass       | Stream         | Local            | Local           | Explainability | Scikit-Learn, NLTK |

## Installation

Install the SDK via pip: an Arthur engineer will provide you with credentials to install the latest version. For example:

`pip install arthurai`. 


#### Using Pipenv
You can also install using the included Pipfile:
1. Export the username of the provided credentials as: `export NEXUS_REPOSITORY_USERNAME=USERNAME`
1. Export the password of the provided credentials as: `export NEXUS_REPOSITORY_PASSWORD=PASSWORD`
1. Run the following command to handle the issue noted above: `export PIP_NO_CACHE_DIR=false`
1. `pipenv install --skip-lock`
1. To enable virtualenv in your shell do `pipenv shell`

#### Passing Credentials To SDK Client:
Credentials can be passed to the SDK client in two ways:
1. Pass via function parameters.
```
from arthurai import ArthurAI
client = ArthurAI(url='app.arthur.ai', access_key='<YOUR-API-KEY>')
```
2. Set SDK client credentials with environment variables. When creating the client with no parameters passed in the sdk will automatically fill credential values with environment variables `ARTHUR_API_KEY` and `ARTHUR_ENDPOINT_URL`.
```bash
$ export ARTHUR_API_KEY="..."
$ export ARTHUR_ENDPOINT_URL="app.arthur.ai"
```
```python
from arthurai import ArthurAI
client = ArthurAI()
```

#### Python Versions and Package Management

The `requirements.txt` files in each of the example project directories (`boston_housing_spark_model`, `credit_risk`, `credit_risk_batch`) support python versions `3.6-3.8`.  If using your own model, or if using a python3 version `3.5` or earlier, you will need to update the `requirements.txt` files with compatible package versions. Specifically, ensure that the `requirements.txt` file has the exact package versions you are using in your local environment.

## Projects
If it's your first time onboarding your model to Arthur, check out some of the [example ML projects](./examples/example_projects) to help inspire you. 

#### Credit Model
 * [Onboarding Guide](./examples/credit_card_default/credit_card_default.ipynb): Jupyter notebook for onboarding a model, then setting up explainability and registering sensitive attriutes for bias monitoring.


## Other Guides
* [Batch Model Quickstart](./examples/credit_risk_batch/notebooks/Quickstart.ipynb): Demo of onboarding and sending inferences with a batch model. 
* [Query Guide](./examples/custom_queries/QueryGuide.ipynb):includes examples of fetching metrics from the platform for further analysis and visualization.
* [Data Viz](./examples/custom_visualizations/DataViz.ipynb) Examples of using built in SDK convenience methods to create visualizations on your Arthur Data.

## Docs
For full detail about the SDK and API, refer to the [Arthur documentation](https://docs.arthur.ai).
