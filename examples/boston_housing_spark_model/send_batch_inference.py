import uuid

from arthurai import ArthurAI
from arthurai import ModelType, InputType, Stage, DataType, ArthurModel
from arthurai.client.apiv2.arthur_explainer import ArthurExplainer
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


spark = SparkSession.builder.appName('app').getOrCreate()

client = ArthurAI(url='app.arthur.ai', access_key='<access_key>')

# lets make inferences on the test data and then send to Arthur
# first lets rename the medv column to be the ground truth column
test = spark.read.parquet("./data/test.parquet")
test = test.withColumnRenamed("medv","medv_ground_truth")

model = client.get_model_by_name("boston_housing_model")

pipeline_input_attr_names = [attr.as_dict()['name'] for attr in model.get_attributes_for_stage(Stage.ModelPipelineInput)]

# load in saved mode pipeline
loaded_model_pipeline = PipelineModel.load("./data/models/boston_housing_spark_model_pipeline")

# make predictions
predicted_dataframe = loaded_model_pipeline.transform(test).withColumnRenamed("prediction", "medv")

# In order to send ground truth we must use an external id to match up rows in the ground truth dataframe and
# inferences dataframe
uuidUdf= udf(lambda : str(uuid.uuid4()), StringType())
predicted_dataframe = predicted_dataframe.withColumn('external_id', uuidUdf())

# Now we separate out the inference input dataframe frame and the ground truth dataframe
pipeline_input_attr_names = [attr.as_dict()['name'] for attr in model.get_attributes_for_stage(Stage.ModelPipelineInput)]
columns_to_select = pipeline_input_attr_names + ['medv', 'external_id']
batch_inferences = predicted_dataframe.select(columns_to_select)

# getting ground truth batch dataframe
columns_to_select = ['medv_ground_truth', 'external_id']
ground_truth_batch = predicted_dataframe.select(columns_to_select)

# write inferences dataframe to parquet file
batch_inferences.write.mode('overwrite').parquet("./data/batch_inference_files/batch_inferences.parquet")
ground_truth_batch.write.mode('overwrite').parquet("./data/batch_ground_truth_files/ground_truth.parquet")

model.send_batch_inferences(directory_path='./data/batch_inference_files/')
model.send_batch_ground_truths(directory_path='./data/batch_ground_truth_files/')
