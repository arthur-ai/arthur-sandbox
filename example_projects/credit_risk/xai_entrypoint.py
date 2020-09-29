import joblib
import os

from pathlib import Path
from model_utils import transformations

model_path = os.path.join(Path(__file__).resolve().parents[0],
                          "fixtures/serialized_models/credit_model.pkl")
sk_model = joblib.load(model_path)


def predict(x):
    return sk_model.predict_proba(x)
