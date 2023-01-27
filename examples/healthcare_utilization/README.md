# Healthcare Utilization

This is a batch model using Arthur's v3 API & SDK, using the Medical Expenditure Panel Survey data.
We use the raw data files and preprocessing provided by [IBM AIF360](https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md).

The prediction task is to determine whether an individual will have *high* utilization
of the healthcare system---that is, visit a medical professional 10 or more times---or
*low* utilization.

To run this notebook, you will need the Arthur SDK installed. Contact an Arthur engineer if
you are having issues with the installation or with the imports at the top of the notebook.

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

The notebook [FullGuide.ipynb](FullGuide.ipynb) shows an example of onboarding a model and sending data.
