import argparse
import joblib
import datetime
import sys
import time
import logging

import numpy as np
from arthurai import ArthurAI, ModelType, InputType, Stage

from model_utils import transformations, load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def send_inferences(access_key, api_url, model_name,
                    training_data_filepath, model_filepath):
    """Send inferences to Arthur platform. After fetching our model by name, we iterate
    over a dataset to get predictions from our sklearn model and then log those inferences."""

    _, (X_test, Y_test) = load_datasets(training_data_filepath)
    sk_model = joblib.load(model_filepath)

    connection = ArthurAI({"access_key": access_key, "url": api_url})
    arthur_model = connection.get_model(model_name)

    for i in range(X_test.shape[0]):
        datarecord = transformations(X_test.iloc[i:i+1, :])
        prediction = sk_model.predict_proba(transformations(datarecord))[0, 1]
        ground_truth = Y_test.iloc[i]
        ext_id = str(np.random.randint(1e9))

        logging.info("Sending inference {}".format(ext_id))
        arthur_model.send_inference(
            inference_timestamp=datetime.datetime.utcnow(),
            external_id=ext_id,
            model_pipeline_input=datarecord.to_dict(orient='records')[0],
            predicted_value=arthur_model.binarize({1: prediction}),
            ground_truth=arthur_model.one_hot_encode(ground_truth)
        )
        time.sleep(np.random.random())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("-u", "--api_url", dest="api_url", required = False,
                        help="The api url", default="dashboard.arthur.ai")
    parser.add_argument("-k", "--access_key", dest="access_key", required=True, help="The api access key")
    parser.add_argument("-n", "--model_name", dest="model_name", required=True, help="Name of model")
    parser.add_argument("-f", "--training_data_filepath", dest="training_data_filepath",
                        required=True, help="Full filepath to training data")
    parser.add_argument("-m", "--model_filepath", dest="model_filepath",
                        required=True, help="Full filepath to serialized model")

    args = parser.parse_args()

    send_inferences(args.access_key, args.api_url, args.model_name,
                    args.training_data_filepath, args.model_filepath)
