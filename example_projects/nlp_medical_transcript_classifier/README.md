# Medical Transcript Classifier

This model is based on the [Kaggle Medical Transcriptions dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions).
It has medical transcripts along with the medical specialty they represent. We will build a classifier that will predict
the medical specialty given the transcription text. While the dataset has thousands of specialties, we limit ourselves to
a subset of 10.


## Setup

Create a virtual environment using your favorite environment management tool and install the requirements. As an example,

```
python3 -m venv env
source env/bin/activate

pip3 install arthurai --index-url https://repository.arthur.ai/repository/pypi-virtual/simple
pip3 install -r medical_transcript_model/requirements.txt
```
Note that the requirements.txt file in this directory assumes python versions 3.6-3.8. If using your own model, or using a python version 3.5 or earlier, you will need to update requirements.txt with compatible package versions. Specifically, ensure that requirements.txt has the exact package versions you are using locally.

## Quickstart

The notebook [notebooks/Quickstart.ipynb](notebooks/Quickstart.ipynb) shows an example of onboarding a model and sending data.

## Other Files

While this repo contains a pre-trained model and everything else you need to get started, the code used to generate the model is included for your reference.

* [nlp_medical_transcript_model/create_model.py](nlp_medical_transcript_model/create_model.py) is the code used to create the model
* [nlp_medical_transcript_model/entrypoint.py](nlp_medical_transcript_model/entrypoint.py) is the code used to enable explainability
* Pickle files are result of running `create_model.py`