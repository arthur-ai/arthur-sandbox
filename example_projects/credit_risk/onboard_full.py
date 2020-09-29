import logging
import argparse
import sys
import os

from pathlib import Path
from arthurai import ArthurAI, ModelType, InputType, Stage
from arthurai.client.apiv2.arthur_explainer import ArthurExplainer

from model_utils import load_datasets

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
MODEL_METADATA = {
            "tags": ["credit"],
            "description": """" RandomForest model trained on 2009 Taiwan Credit Default dataset. 
        https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        """,
            "input_type": InputType.Tabular,
            "model_type": ModelType.Multiclass
        }


def onboard_model(access_key: str, api_url: str,
                  model_name: str, training_data_filepath: str) -> None:
    """Example of onboarding a Tabular classification model and also enabling bias
     monitoring for several sensitive attributes as well as enabling explainability."""

    MODEL_METADATA['name'] = model_name
    (X_train, Y_train), _ = load_datasets(training_data_filepath)

    connection = ArthurAI({"access_key": access_key, "url": api_url})
    arthur_model = connection.model(**MODEL_METADATA)

    # Set up model basics
    logging.info("Setting data schema")
    arthur_model.from_dataframe(X_train, Stage.ModelPipelineInput)
    arthur_model.from_dataframe(Y_train, Stage.GroundTruth)
    arthur_model.set_positive_class(1)

    # Set up bias monitoring for sensitive attributes
    arthur_model.get_attribute("SEX",
                        stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("EDUCATION",
                        stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("AGE",
                        stage=Stage.ModelPipelineInput).monitor_for_bias = True
    arthur_model.get_attribute("AGE",
                        stage=Stage.ModelPipelineInput).cutoffs = [35, 55]

    # Supply readable field names for categorical variables
    arthur_model.set_attribute_labels("SEX",
                               Stage.ModelPipelineInput,
                               labels={1: "Male", 2: "Female"})
    arthur_model.set_attribute_labels("EDUCATION",
                               Stage.ModelPipelineInput,
                               labels={1: "Graduate School", 2: "University",
                                       3: "High School", 4: "Less Than High School",
                                       5: "Unknown", 6: "Unreported", 0: "Other"})
    arthur_model.set_attribute_labels("MARRIAGE",
                               Stage.ModelPipelineInput,
                               labels={1: "Married", 2: "Single",
                                       3: "Other", 0: "Unknown"})
    arthur_model.set_attribute_labels("default payment next month",
                               Stage.GroundTruth,
                               labels={0: "Creditworthy",
                                       1: "CreditDefault"})

    logging.info("Enabling explainability")
    path = Path(__file__).resolve()
    arthur_model.enable_explainability(
        df=X_train.head(50),
        project_directory=path.parents[0],
        requirements_file="requirements.txt",
        user_predict_function_import_path="xai_entrypoint",
        explanation_algo=ArthurExplainer.SHAP)

    logging.info("Saving model")
    arthur_model.save()

    logging.info("Setting reference data")
    # Note - this step is optional. If you don't upload a reference set, Arthur
    # will use the first 5000 inferences to set the baseline.
    arthur_model.set_reference_data(stage=Stage.ModelPipelineInput, data=X_train)

if __name__== "__main__":
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("-u", "--api_url", dest="api_url", required = False,
                        help="The api url", default="dashboard.arthur.ai")
    parser.add_argument("-k", "--access_key", dest="access_key", required=True, help="The api access key")
    parser.add_argument("-n", "--model_name", dest="model_name", required=True, help="Name of model")
    parser.add_argument("-f", "--training_data_filepath", dest="training_data_filepath",
                        required=True, help="Full filepath to training data")
    args = parser.parse_args()

    onboard_model(args.access_key, args.api_url, args.model_name, args.training_data_filepath)
