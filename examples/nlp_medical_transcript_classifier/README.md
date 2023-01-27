# Medical Transcript Classifier

This model is based on the [Kaggle Medical Transcriptions dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions).
It has medical transcripts along with the medical specialty they represent. We will build a classifier that will predict
the medical specialty given the transcription text. While the dataset has thousands of specialties, we limit ourselves to
a subset of 10.


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

The notebook [notebooks/Quickstart.ipynb](notebooks/Quickstart.ipynb) shows an example of onboarding a model and sending data.

## Other Files

While this repo contains a pre-trained model and everything else you need to get started, the code used to generate the model is included for your reference.

* [create_model.py](create_model.py) is the code used to create the model
* [entrypoint.py](entrypoint.py) is the code used to enable explainability
* Pickle files are result of running `create_model.py`
