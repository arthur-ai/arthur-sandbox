[![image](https://arthurai-assets.s3.amazonaws.com/logo-black-transparent.png?format=300w)](https://arthur.ai)
# Getting Started with Arthur

In this repository, you'll find sample projects of integrating machine learning models with [Arthur](https://www.arthur.ai/), 
with tips on how to get the most out of the Arthur Python SDK. 

Data scientists and machine learning practitioners use Arthur to get insights about their models' performance and 
centralize their organization's use of AI in one unified system for tracking and analysis. You can find more information 
about the Arthur SDK and API on our [documentation site](https://docs.arthur.ai). 

## Quickstart

**Check out the [Credit Card Default example project](./examples/credit_card_default) for a beginner-friendly example.**  
It walks through onboarding a tabular binary classification model to the Arthur platform and showcases the 
full suite of Arthur features.

## Installing the SDK

To install the Arthur Python SDK, you can use the `pip` package manager, and run `pip install arthurai` at your command 
line.

## Arthur Sandbox: Examples of Integrating with Arthur

Each example shows you how to prepare data, register models with Arthur, send inferences to the platform, and use 
Arthur features to analyze model performance. The examples cover a variety of data input types, modeling tasks, and
Arthur features showcased, but all have the same basic model onboarding workflow. For example, Arthur 
offers monitoring services for models used in [Computer Vision](./examples/cv_mars_rover/Quickstart.ipynb) (CV) and 
[Natural Language Processing](./examples/nlp_medical_transcript_classifier/notebooks/Quickstart.ipynb) (NLP) 
applications. See the table below for a reference of examples in this repository.

### Examples Directory

| Example             | Input Type  | Output Type      | Inference Type | Data Storage     | Model Storage   | Arthur Features | Frameworks |
| ------------------- | ----------- | ---------------- | -------------- | ---------------- | --------------- | --------------- | ---------- |
| [Credit Card Default](./examples/credit_card_default/credit_card_default.ipynb)         | Tabular     | Binary           | Batch & stream | Local            | Local           | Performance, Drift, Visualization, Alerts, Bias, Explainability, Anomaly Detection, Hotspots, Bias Mitigation | Scikit-Learn    |
| [MEPS Healthcare](./examples/healthcare_utilization/FullGuide.ipynb)     | Tabular     | Binary           | Stream         | Local            | Local           | Explainability | Scikit-Learn, NLTK |
| [Boston Housing](./examples/boston_housing_spark_model/FullGuide.ipynb)      | Tabular     | Regression       | Batch          | Local            | Local           | Explainability | Spark ML |
| [Cancer Detection](./examples/cv_cancer_detection/FullGuide.ipynb)    | Image       | Binary           | Stream         | Arthur S3 Bucket | Local           | Explainability | PyTorch |
| [Mars Rover](./examples/cv_mars_rover/Quickstart.ipynb)          | Image       | Object Detection | Stream         | Nasa API and Arthur S3 Bucket        | SageMaker           | Explainability | Scikit-Learn, OpenCV, PyTorch |
| [Object Detection](./examples/cv_object_detection/Quickstart.ipynb)    | Image       | Object Detection | Stream         | Arthur S3 Bucket | Local           | Explainability | OpenCV, TensorFlow |
| [Satellite Images](./examples/cv_satellite_country_prediction/FullGuide.ipynb)    | Image       | Binary           | Batch          | Arthur S3 Bucket | Local           | Explainability | PyTorch |
| [Medical Transcripts](./examples/nlp_medical_transcript_classifier/notebooks/Quickstart.ipynb) | NLP         | Multiclass       | Stream         | Local            | Local           | Explainability | Scikit-Learn, NLTK |
