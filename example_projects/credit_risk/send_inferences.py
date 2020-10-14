import argparse
import joblib
import datetime
import sys
import time
import logging

import numpy as np
from arthurai import ArthurAI

from model_utils import load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def send_inferences(access_key, api_url, model_name,
                    training_data_filepath, model_filepath):
    """Send inferences to Arthur platform. After fetching our model by name, we iterate
    over a dataset to get predictions from our sklearn model and then log those inferences."""

    _, (X_test, Y_test) = load_datasets(training_data_filepath)
    sk_model = joblib.load(model_filepath)

    connection = ArthurAI(url=api_url, access_key=access_key, client_version=3)
    arthur_model = connection.get_model(identifier=model_name, id_type="partner_model_id")

    for i in range(X_test.shape[0]):
        datarecord = X_test.iloc[i:i+1, :]
        predicted_probs = sk_model.predict_proba(datarecord)[0]
        ground_truth = np.int(Y_test.iloc[i])
        ext_id = str(np.random.randint(1e9))

        logging.info("Sending inference {}".format(ext_id))
        arthur_model.send_inference(
            inference_timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
            external_id=ext_id,
            model_pipeline_input=datarecord.to_dict(orient='records')[0],
            predicted_value={"prediction_1":predicted_probs[1],
                             "prediction_0":predicted_probs[0]},
            ground_truth={"gt_1": ground_truth,
                          "gt_0":1-ground_truth}
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
