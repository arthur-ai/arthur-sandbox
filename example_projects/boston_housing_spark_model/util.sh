#!/bin/bash

cmd=$1
shift
echo "@"

case $cmd in 
    'build') docker build --build-arg $1 --build-arg $2 -t pyspark-notebook -f Dockerfile .
    ;;
    'run-docker') docker run -it --rm -v $PWD:/app -w /app pyspark-notebook python3 "$@"
    ;;
    'notebook') docker run -it --rm -p 8888:8888 -v $PWD:/app -w /app pyspark-notebook
    ;;
    '') echo "Usage: $0 <command>"; exit 1
    ;;
    *) echo "Unknown command: $cmd"; exit 1
    ;;
esac