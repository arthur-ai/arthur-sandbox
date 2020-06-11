# Boston Housing SparkML Model

This example model uses the [Boston Housing Datset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to build a sparkML linear regressor model. Then it walks through the steps to on-board this model to Arthur including the steps required to enable the explainability pipeline specifically for sparkML models. All of the required steps can be viewed in the notebook `create_and_onboard.ipynb` or in python files which break out the process into three steps, `create_model.py`, `upload_model.py`, and `send_batch_inference.py`.

### Setup 

Create a virtual environment using your favorite environment management tool and install the requirements. Separately ensure the latest version of ArthurAI SDK is installed. As an example,

```commandline
python3 -m venv env
source env/bin/activate

pip3 install -r requirements.txt
```

If you do not have pyspark installed locally you can use the provided Dockerfile. The follow util commands will run the python and notebook files within docker:
<br>
To run the notebook in the docker container
```commandline
./util.sh build
./util.sh notebook
```

To run the python scripts in the docker container:
```commandline
./util.sh build
./util.sh [create_model.py, upload_model.py, send_batch_inference.py]
```

#### Enable Explainability For a SparkML Model:
The steps for uploading the model and sending batch inferences are the same regardless of what type of machine learning model is being used. There are however a few unique steps that must be taken to on-board a SparkML model with explainability. The first step is to create an entrypoint that loads in the SparkML model and implements the `predict()` function. Unique for a Spark model this file must also initialize a pyspark session in local mode:
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').appName('app').getOrCreate()
```
Optionally the spark session can be configured to use a specified amount of cpu or memory. Whatever configuration is specified in the entrypoint file must also be used to initialize the model server using the sdk's function `ArthurModel.enable_explainability(...)`. An example can be seen below:
```python
model.enable_explainability(df=train_df, project_directory='.',
                            user_predict_function_import_path='entrypoint',
                            requirements_file='requirements.txt',
                            model_server_num_cpu='2',
                            model_server_memory='500Mi')
```
**Note, the minimum amount of cpus which must be allocated for spark models is 2**. This must explicitly be defined in the call to `enable_explainability(...)`.
The full `entrypoint.py` example file is provided in this repo. Once these steps are taken and `ArthurModel.save()` is called, the model will be uploaded and intialized with explainability.
