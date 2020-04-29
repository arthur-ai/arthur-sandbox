
# Credit Risk

This example model is based on the [UCI Taiwan Credit dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). With this dataset, we build a binary classifier for predicting credit default risk from various attributes about payment information and customer demographics. Since this dataset includes  attributes such as Age and Gender, we can also use this example to take a close look at algorithmic fairness. 

### Setup 

Create a virtual environment using your favorite environment management tool and install the requirements. As an example,

```commandline
python3 -m venv env
source env/bin/activate

pip3 install -r requirements.txt
```

#### Training a model
We'll train a random forest classifier on this dataset. This step is entirely optional, since a pre-trained classifier is already in `fixtures/serialized_models/credit_model.pkl`.

```commandline
python3 train.py  -f ./fixtures/datasets/credit_card_default.csv -m ./fixtures/serialized_models/credit_model.pkl 
```

#### Onboarding a Model
Examples of how to onboard a model can be found in [`onboard_quick.py`](./onboard_quick.py) and [`onboard_full.py`](./onboard_full.py). Additionally, the notebook [`notebooks/Quickstart.ipynb`](./notebooks/Quickstart.ipynb) shows an example of onboarding a model.

*Onboard Quick*

To get a quick understanding of how to onboard a model to the platform, [`onboard_quick.py`](./onboard_quick.py) demonstrates how to get started. Just supply your API key, a model name, and a pointer to an example of the training data.
```commandline
python onboard_quick.py --access_key $ACCESS_KEY  --model_name testy_mctestface_1.0.1 --training_data_filepath ./fixtures/datasets/credit_card_default.csv 
```

*Onboard Full*

To get a better sense of the full features available in Arthur, [`onboard_full.py`](./onboard_full.py) shows an example of onboarding the same type of model while also enabling Bias monitoring and Explainability. 
```commandline
python sonboard_full.py --access_key $ACCESS_KEY  --model_name testy_mctestface_1.0.2 --training_data_filepath ./fixtures/datasets/credit_card_default.csv 
```

Once you've onboarded a model, log in to the Arthur dashboard to verify that a new model is created and ready to receive data.


#### Sending Inferences
Once your model has been onboarded, you can begin sending inferences. 

```commandline
python send_inferences.py --access_key $ACCESS_KEY --model_name testy_mctestface_1.0.2 --training_data_filepath ./fixtures/datasets/credit_card_default.csv --model_filepath ./fixtures/serialized_models/credit_model.pkl
```

After you've sent some inferences, log in to the Arthur dashboard to explore metrics, drift, explanations, and fairness.