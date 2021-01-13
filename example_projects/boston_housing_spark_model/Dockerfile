ARG BASE_CONTAINER=openjdk:11-buster
FROM $BASE_CONTAINER

ARG NEXUS_REPOSITORY_USERNAME
ARG NEXUS_REPOSITORY_PASSWORD

# Spark dependencies
ENV APACHE_SPARK_VERSION=3.0.1 \
    HADOOP_VERSION=2.7

# Using the preferred mirror to download Spark
RUN cd /tmp && \
    wget -q https://mirrors.gigenet.com/apache/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local --owner root --group root --no-same-owner && \
    rm spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

# Create notebook user
RUN useradd -ms /bin/sh notebook
USER notebook
ENV PATH="/home/notebook/.local/bin:${PATH}"
WORKDIR /home/notebook

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
# manually install llvmlite from conda
RUN ./miniconda/bin/conda install -y python==3.7.9 && \
    ./miniconda/bin/conda install -c conda-forge -y shap==0.37.0

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip \
    SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin \
    PYSPARK_PYTHON=/home/notebook/miniconda/bin/python \
    PYSPARK_DRIVER_PYTHON=/home/notebook/miniconda/bin/python

# add project specific files
WORKDIR /app

# check that PyPI args exist
RUN test -n "$NEXUS_REPOSITORY_USERNAME" && test -n "$NEXUS_REPOSITORY_PASSWORD"
RUN ~/miniconda/bin/conda run python3 -m pip install --no-cache --extra-index-url https://${NEXUS_REPOSITORY_USERNAME}:${NEXUS_REPOSITORY_PASSWORD}@repository.arthur.ai/repository/pypi-virtual/simple arthurai

COPY . .
RUN ~/miniconda/bin/conda run python3 -m pip install --no-cache-dir -r requirements.txt 

ENTRYPOINT ~/miniconda/bin/conda run --no-capture-output jupyter notebook --no-browser --ip 0.0.0.0
