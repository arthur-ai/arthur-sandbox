
# Credit Risk

This example model is based on the [UCI Taiwan Credit dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). With this dataset, we build a binary classifier for predicting credit default risk from various attributes about payment information and customer demographics. Since this dataset includes  attributes such as Age and Gender, we can also use this example to take a close look at algorithmic fairness. 

### Setup 

Create a virtual environment using your favorite environment management tool and install the requirements. As an example,

```commandline
python3 -m venv env
source env/bin/activate

pip3 install arthurai --index-url https://repository.arthur.ai/repository/pypi-virtual/simple
pip3 install -r requirements.txt
```


Note that the `requirements.txt` file in this directory assumes python versions `3.6-3.8`. If using your own model, or using a python version `3.5` or earlier, you will need to update `requirements.txt` with compatible package versions. Specifically, ensure that `requirements.txt` has the exact package versions you are using locally.

### Quickstart 
The notebook [`notebooks/Quickstart.ipynb`](./notebooks/Quickstart.ipynb) shows an example of onboarding a model and sending data. We recommend you start here as you familiarize yourself with the Arthur platform.

After you've looked at the Quickstart, you can look at each component in more depth below:

#### Training a model

We'll train a random forest classifier on this dataset. This step is entirely optional, since a pre-trained classifier is already in `fixtures/serialized_models/credit_model.pkl`.

```commandline
python3 train.py  -f ./fixtures/datasets/credit_card_default.csv -m ./fixtures/serialized_models/credit_model.pkl 
```

#### Onboarding a Model
Examples of how to onboard a model can be found in [`onboard.py`](./onboard.py). Additionally, the notebook [`notebooks/Quickstart.ipynb`](./notebooks/Quickstart.ipynb) shows an example of onboarding a model.

*Onboard*

[`onboard.py`](./onboard.py) shows an example of onboarding a model while also enabling Bias monitoring and Explainability. 
```commandline
python onboard.py --access_key $ACCESS_KEY  --model_name testy_mctestface_1.0.2 --training_data_filepath ./fixtures/datasets/credit_card_default.csv 
```

Once you've onboarded a model, log in to the Arthur dashboard to verify that a new model is created and ready to receive data.


#### Sending Inferences
Once your model has been onboarded, you can begin sending inferences. 

```commandline
python send_inferences.py --access_key $ACCESS_KEY --model_name testy_mctestface_1.0.2 --training_data_filepath ./fixtures/datasets/credit_card_default.csv --model_filepath ./fixtures/serialized_models/credit_model.pkl
```

After you've sent some inferences, log in to the Arthur dashboard to explore metrics, drift, explanations, and fairness.
