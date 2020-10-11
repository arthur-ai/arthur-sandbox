from arthurai import ArthurAI
from arthurai.client.apiv3 import OutputType, InputType, Stage, ValueType, AttributeCategory
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('app').getOrCreate()

client = ArthurAI(url='app.arthur.ai', access_key='<access_key>')

# create a dataframe from the training data to on-board model metadata
train = spark.read.parquet("./data/train.parquet")
train_df = train.toPandas()
train_df = train_df.drop('medv', axis=1)  # drop predicted value column to leave only pipeline input


# create ArthurModel
MODEL_METADATA = {
    "name": 'boston_housing_model',
    "description": "Spark Boston Housing Model",
    "input_type": InputType.Tabular,
    "model_type": OutputType.Regression,
    "tags": ['Spark'],
    "is_batch": True
}

model = client.model(**MODEL_METADATA)
model.from_dataframe(train_df[list(train_df.columns)[0:]], Stage.ModelPipelineInput)
model.add_attribute(
    name='medv_gt',
    stage=Stage.GroundTruth,
    data_type=DataType.Float,
    categorical=False,
    attribute_link="medv_prediction"
)

model.save()

model.enable_explainability(df=train_df, project_directory='.',
                            user_predict_function_import_path='entrypoint',
                            requirements_file='requirements.txt',
                            model_server_num_cpu='2', model_server_memory='1Gi')