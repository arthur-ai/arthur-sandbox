`serialized_models` contains pkl files for 3 different model trained on the UCI Taiwan Credit dataset, found in CSV form in `datasets`.
The data used here is the exact same as the credit risk data.
The CSV has 30k rows; the model is trained on the first 5k and tested (in the notebook) on the last 5k.

Three different models:

1. `credit_lr.pkl` Scikit-learn logistic regression
2. `credit_rf.pkl` Scikit-learn random forest
3. `credit_frf.pkl` Fairlearn + Scikit-learn "fair" random forest

All models implement `.predict(x)` and `predict_proba(x)`. `x` should be values, not dataframe.