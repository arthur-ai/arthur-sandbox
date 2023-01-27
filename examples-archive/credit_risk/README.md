# Credit Risk

This example model is based on the [UCI Taiwan Credit dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). With this dataset, we build a binary classifier for predicting credit default risk from various attributes about payment information and customer demographics. Since this dataset includes  attributes such as Age and Gender, we can also use this example to take a close look at algorithmic fairness. 

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
The notebook [notebooks/Quickstart.ipynb](notebooks/Quickstart.ipynb) provides a walkthrough of how to onboard streaming models and send inferences.

## Detailed Guide
The notebook [notebooks/FullGuide.ipynb](notebooks/FullGuide.ipynb) provides examples of enables additional functionality such as Bias monitoring and Explainability. 