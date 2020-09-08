import argparse
import logging
import joblib
import sys

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from model_utils import load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def fit_model(training_data_filepath, model_output_path):
    (X_train, Y_train), (X_test, Y_test) = load_datasets(training_data_filepath)
    model = RandomForestClassifier(n_estimators=500, max_depth=15)
    logging.info(" Beginning model fitting...")
    model.fit(X_train, Y_train)

    logging.info(" Test Error for model {:.3f}".format(
        model.score(X_test, Y_test)))

    logging.info(" Saving model to {}".format(Path(model_output_path).resolve()))
    joblib.dump(model, model_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("-f", "--training_data_filepath", dest="training_data_filepath",
                        required=True, help="Full filepath to training data")
    parser.add_argument("-m", "--model_output_path", dest="model_output_path", required=True,
                        help="Full filepath with extension to save the final model")
    args = parser.parse_args()

    fit_model(args.training_data_filepath, args.model_output_path)
