import logging
import argparse
import sys
import joblib

from pathlib import Path
from arthurai import ArthurAI
from arthurai.common.constants import InputType, OutputType, Stage
from arthurai.explainability.arthur_explainer import ArthurExplainer

from model_utils import load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def onboard_model(access_key: str, api_url: str,
                  model_name: str, training_data_filepath: str) -> None:
    """Example of onboarding a Tabular classification model and also enabling bias
     monitoring for several sensitive attributes as well as enabling explainability."""

    (X_train, Y_train), _ = load_datasets(training_data_filepath)

    connection = ArthurAI(url=api_url, access_key=access_key)
    arthur_model = connection.model(partner_model_id=model_name,
                                    input_type=InputType.Tabular,
                                    output_type=OutputType.Multiclass)

    # Set up model basics
    logging.info("Setting data schema")
    arthur_model.from_dataframe(X_train, Stage.ModelPipelineInput)
    prediction_to_ground_truth_map = {
        "prediction_0": "gt_0",
        "prediction_1": "gt_1"
    }
    arthur_model.add_binary_classifier_output_attributes("prediction_1", prediction_to_ground_truth_map)

    # Set up bias monitoring for sensitive attributes
    arthur_model.get_attribute("SEX", stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("EDUCATION", stage=Stage.ModelPipelineInput).monitor_for_bias = True

    logging.info("Saving model")
    arthur_model.save()

    logging.info("Enabling explainability")
    path = Path(__file__).resolve()
    arthur_model.enable_explainability(
        df=X_train.head(50),
        project_directory=path.parents[0],
        requirements_file="requirements.txt",
        user_predict_function_import_path="xai_entrypoint",
        streaming_explainability_enabled=True,
        explanation_algo=ArthurExplainer.SHAP,
        model_server_memory="2G",
        model_server_num_cpu="2"
    )

    logging.info("Setting reference data")
    # load our pre-trained classifier so we can generate predictions
    sk_model = joblib.load("fixtures/serialized_models/credit_model.pkl")

    # get all input columns
    reference_set = X_train.copy()

    # get ground truth labels
    reference_set["gt_1"] = Y_train
    reference_set["gt_0"] = 1 - Y_train

    # get model predictions
    preds = sk_model.predict_proba(X_train)
    reference_set["prediction_1"] = preds[:, 1]
    reference_set["prediction_0"] = preds[:, 0]

    arthur_model.set_reference_data(data=reference_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-u", "--api_url", dest="api_url", required=False,
                        help="The api url", default="https://app.arthur.ai")
    parser.add_argument("-k", "--access_key", dest="access_key", required=True, help="The api access key")
    parser.add_argument("-n", "--model_name", dest="model_name", required=True, help="Name of model")
    parser.add_argument("-f", "--training_data_filepath", dest="training_data_filepath",
                        required=True, help="Full filepath to training data")
    args = parser.parse_args()

    onboard_model(args.access_key, args.api_url, args.model_name, args.training_data_filepath)
