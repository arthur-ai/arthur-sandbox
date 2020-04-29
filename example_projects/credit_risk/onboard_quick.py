import argparse

from arthurai import ArthurAI, ModelType, InputType, Stage
from model_utils import  load_datasets

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
    """Example of onboarding a Tabular classification model."""

    MODEL_METADATA['name'] = model_name
    (X_train, Y_train), _ = load_datasets(training_data_filepath)

    connection = ArthurAI({"access_key": access_key, "url": api_url})
    arthur_model = connection.model(**MODEL_METADATA)

    arthur_model.from_dataframe(X_train, Stage.ModelPipelineInput)
    arthur_model.from_dataframe(Y_train, Stage.GroundTruth)
    arthur_model.set_positive_class(1)

    arthur_model.save()


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
