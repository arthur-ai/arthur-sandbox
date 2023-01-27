FROM python:3.9-bullseye

ENV JUPYTER_ENABLE_LAB yes
ENV PYTHONUNBUFFERED TRUE

# install libGL
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install arthurai
COPY deploy/sagemaker_requirements.txt /tmp/sagemaker_requirements.txt
RUN pip install -r  /tmp/sagemaker_requirements.txt

# create "python3" kernel

ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY ./deploy/run_notebook ./deploy/execute.py /opt/program/
ENTRYPOINT ["/bin/bash"]

# because there is a bug where you have to be root to access the directories
USER root

