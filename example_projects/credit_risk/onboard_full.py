import logging
import argparse
import sys
import os

from pathlib import Path
from arthurai import ArthurAI
from arthurai.common.constants import OutputType, InputType, Stage
from arthurai.core.attributes import AttributeCategory
from arthurai.explainability.arthur_explainer import ArthurExplainer

from model_utils import load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
MODEL_METADATA = {
            "tags": ["credit"],
            "description": """" RandomForest model trained on 2009 Taiwan Credit Default dataset. 
        https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        """,
            "input_type": InputType.Tabular,
            "model_type": OutputType.Multiclass
        }


def onboard_model(access_key: str, api_url: str,
                  model_name: str, training_data_filepath: str) -> None:
    """Example of onboarding a Tabular classification model and also enabling bias
     monitoring for several sensitive attributes as well as enabling explainability."""

    MODEL_METADATA['partner_model_id'] = model_name
    (X_train, Y_train), _ = load_datasets(training_data_filepath)

    connection = ArthurAI({"access_key": access_key, "url": api_url})
    arthur_model = connection.model(**MODEL_METADATA)

    # Set up model basics
    logging.info("Setting data schema")
    arthur_model.from_dataframe(X_train, Stage.ModelPipelineInput)
    prediction_to_ground_truth_map = {
        "prediction_0": "gt_0",
        "prediction_1": "gt_1"
    }
    arthur_model.add_binary_classifier_output_attributes("prediction_1", prediction_to_ground_truth_map)

    # Set up bias monitoring for sensitive attributes
    sex_attr = arthur_model.get_attribute("SEX", stage=Stage.ModelPipelineInput)
    sex_attr.monitor_for_bias = True
    arthur_model.get_attribute("EDUCATION", stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("AGE",stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("AGE", stage=Stage.ModelPipelineInput).cutoffs = [35, 55]

    # Supply readable field names for categorical variables
    arthur_model.set_attribute_labels(attribute_name="SEX",
                               labels={1: "Male", 2: "Female"})
    arthur_model.set_attribute_labels(attribute_name="EDUCATION",
                               labels={1: "Graduate School", 2: "University",
                                       3: "High School", 4: "Less Than High School",
                                       5: "Unknown", 6: "Unreported", 0: "Other"})
    arthur_model.set_attribute_labels(attribute_name="MARRIAGE",
                               labels={1: "Married", 2: "Single",
                                       3: "Other", 0: "Unknown"})

    logging.info("Saving model")
    arthur_model.save()

    logging.info("Setting reference data")
    arthur_model.set_reference_data(data=X_train)

    logging.info("Enabling explainability")
    path = Path(__file__).resolve()
    arthur_model.enable_explainability(
        df=X_train.head(50),
        project_directory=path.parents[0],
        requirements_file="requirements.txt",
        user_predict_function_import_path="xai_entrypoint",
        explanation_algo=ArthurExplainer.SHAP,
        streaming_explainability_enabled=True)


if __name__== "__main__":
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("-u", "--api_url", dest="api_url", required = False,
                        help="The api url", default="https://app.arthur.ai")
    parser.add_argument("-k", "--access_key", dest="access_key", required=True, help="The api access key")
    parser.add_argument("-n", "--model_name", dest="model_name", required=True, help="Name of model")
    parser.add_argument("-f", "--training_data_filepath", dest="training_data_filepath",
                        required=True, help="Full filepath to training data")
    args = parser.parse_args()

    onboard_model(args.access_key, args.api_url, args.model_name, args.training_data_filepath)
