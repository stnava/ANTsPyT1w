#!/bin/bash

container_name="$1"
script_path=$2
config_path=$3
antspy_hash="e7e8644857a78c442aff5e688ccd491164746b24"
antspynet_hash="02da39786fc92b9903a0c3bdcf40622e003f5622"
superiq_hash=$(git rev-parse HEAD)
aws_profile=${4:-default}

docker build \
    --build-arg AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile "$aws_profile")\
    --build-arg AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile "$aws_profile")\
    --build-arg antspy_hash=$antspy_hash \
    --build-arg antspynet_hash=$antspynet_hash \
    --build-arg superiq_hash=$superiq_hash \
    -t $container_name .

docker run --rm -it \
    --name $container_name \
    -e cpu_threads="8" \
    $container_name:latest \
    python3 $2 $3
