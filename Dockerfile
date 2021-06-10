FROM python:3.7-slim-buster
LABEL maintainer="tgosselin"

RUN apt-get update && \
    apt-get install -y build-essential cmake libpng-dev pkg-config git

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN pip install numpy keras boto3
RUN pip install --upgrade tensorflow tensorflow-probability

ARG antspy_hash
RUN pip install git+https://github.com/ANTsX/ANTsPy.git@$antspy_hash

ARG antspynet_hash
RUN pip install git+https://github.com/ANTsX/ANTsPyNet.git@$antspynet_hash

COPY ia-batch-utils ia-batch-utils
RUN pip install ia-batch-utils/.

COPY setup.py src/setup.py
COPY deploy src/deploy
COPY configs src/configs
COPY superiq src/superiq
WORKDIR src
RUN python setup.py install
