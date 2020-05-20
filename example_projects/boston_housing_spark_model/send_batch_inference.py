from arthurai import ArthurAI
from arthurai import ModelType, InputType, Stage, DataType, ArthurModel
from arthurai.client.apiv2.arthur_explainer import ArthurExplainer
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


spark = SparkSession.builder.appName('app').getOrCreate()

client = ArthurAI(url='dashboard.arthur.ai', access_key='<access_key>')

# lets make inferences on the test data and then send to Arthur
# first lets rename the medv column to be the ground truth column
test = spark.read.parquet("./data/test.parquet")
test = test.withColumnRenamed("medv","medv_ground_truth")

model = client.get_model_by_name("boston_housing_model")

pipeline_input_attr_names = [attr.as_dict()['name'] for attr in model.get_attributes_for_stage(Stage.ModelPipelineInput)]

# load in saved mode pipeline
loaded_model_pipeline = PipelineModel.load("./data/models/boton_housing_spark_model_pipeline")

# make predictions
predicted_dataframe = loaded_model_pipeline.transform(test).withColumnRenamed("prediction", "medv")
columns_to_select = pipeline_input_attr_names + ['medv', 'medv_ground_truth']
predicted_dataframe = predicted_dataframe.select(columns_to_select)

# write inferences dataframe to parquet file
pd_df = predicted_dataframe.toPandas()
pd_df.to_parquet("./data/batch_inference_files/inferences.parquet")

model.send_batch_inferences(directory_path='./data/batch_inference_files/')
