from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('app').getOrCreate()

# load in training and testing data
data = spark.read.csv('./data/boston_housing.csv', header=True, inferSchema=True)
train, test = data.randomSplit([0.7, 0.3])
# save training and testing data
test.write.mode('overwrite').parquet("./data/test.parquet")
train.write.mode('overwrite').parquet("./data/train.parquet")

# create assmbler and model
feature_columns = data.columns[:-1] # here we omit the final column
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
algo = LinearRegression(featuresCol="features", labelCol="medv", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# create the pipeline from a VectorAssembler and LinearRegression model, then train it on the training data partition
pipeline = Pipeline(stages=[assembler, algo]) 
fitted_pipeline = pipeline.fit(train)

# save the pipeline
fitted_pipeline.write().overwrite().save('./data/models/boton_housing_spark_model_pipeline')
