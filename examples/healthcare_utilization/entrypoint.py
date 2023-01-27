import pandas as pd
import joblib
import os

model = joblib.load(f"{os.path.dirname(__file__)}/saved_model/skl_rf.joblib")

def predict(input_data):
    return model.predict_proba(input_data)[:,1]
