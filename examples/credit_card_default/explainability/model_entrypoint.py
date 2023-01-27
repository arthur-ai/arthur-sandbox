# model_entrypoint.py
import joblib

model = joblib.load("credit_model.pkl")

def predict(x):
    return model.predict_proba(x)[:,1]
