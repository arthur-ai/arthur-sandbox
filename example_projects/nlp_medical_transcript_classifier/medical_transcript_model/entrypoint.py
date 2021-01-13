import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

def predict(fvs):
    # our model expects a list of strings, no nesting
    # if we receive nested lists, unnest them
    if not isinstance(fvs[0], str):
        fvs = [fv[0] for fv in fvs]
    return model.predict_proba(fvs)