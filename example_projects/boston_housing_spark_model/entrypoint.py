import pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
import pyspark


# To start the spark session on the model server specify the master url as local.
# By default this will run spark using 1 thread, to increase threads you can specify
# local[x] where x is the number of threads. When allocating more compute and memory to the spark
# session be sure to increase the amount allocated to the model server when calling ArthurModel.enable_explainability()
# in the sdk (by default 1 cpu and 1gb of memory is allocated to the model server).
spark = SparkSession.builder.master('local').appName('app').getOrCreate()

loaded_pipeline = PipelineModel.load("./data/models/boton_housing_spark_model_pipeline")


def predict(input_data):
	col_names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','bb','lstat']
	input_df = pd.DataFrame(input_data, columns=col_names)
	spark_df = spark.createDataFrame(input_df)
	predictions = loaded_pipeline.transform(spark_df)
	return np.array([float(x.prediction) for x in predictions.select('prediction').collect()])