

## Curiosity Mars Rover Terrain Detection

This project demonstrates integrating an object detection model for Mars rover images with the Arthur platform.
You can download and use static datasets, or fetch recent images from the Curiosity rover and even deploy 
your model onto Amazon SageMaker to regularly update the model with a fresh stream of images.

### Quick Start

To jump right in, just fire up the Quickstart notebook in this folder!

### The Data

The training dataset is built from NASA's [AI4Mars Dataset](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix) 
which uses labeled images from the Curiosity, Spirit, and Opportunity rovers. The images for inference come from NASA's 
[Mars Rover Photos API](https://api.nasa.gov/#mars-rover-photos). The segmentation labels on the images belong to four 
possible classes:
- sand
- soil
- bedrock
- big rock

### The Model

This project uses the PyTorch implementation of the [YOLOv5 model](https://github.com/ultralytics/yolov5) to conduct 
the object detection task. Because the AI4Mars dataset uses segmentation labels but the YOLO model provides bounding 
box outputs, the data is transformed to the bounding box format and cleaned. Please see [You Only Look Once: Unified, 
Real-Time Object Detection](https://arxiv.org/abs/1506.02640) for the original YOLO model architecture.

### Directory Structure

The `predict.py` file contains the model prediction logic, and the `util` package in the root folder contains helper 
functions for downloading the model and data, interacting with the NASA API, transforming data, and more. The Quickstart 
notebook walks through loading the data and model, making predictions, and onboarding the model to Arthur.

There are two additional subfolders of note:

#### Training

The `training` folder contains all the code needed to build the model yourself. The "AI4Mars Data Preprocessing" 
notebook walks through downloading the original NASA AI4Mars dataset, converting the segmentation masks into bounding 
boxes, and cleaning the data. The "AI4Mars Model Training" notebook contains a simple command to train the YOLOv5 model.

#### Deploy

The `deploy` folder contains code to deploy the Quickstart notebook to SageMaker Processing. This is adapted from the 
[SageMaker Run Notebook](https://github.com/aws-samples/sagemaker-run-notebook) project. Step through the "SageMaker 
Deployment" notebook to create infrastructure in your AWS account to update your Arthur model with a fresh stream of 
images from the NASA API on a daily basis.

#### Note on Data Downloads

Data is downloaded to the `api-data-download` folder and then copied to the `api-data` folder, creating two copies of 
each image downloaded. This is because SageMaker inputs and ouptuts must be unique, so in AWS the download folder is 
ephemeral while the `api-data` folder always contains the full set of images that have ever been downloaded, in your 
specified S3 bucket.
