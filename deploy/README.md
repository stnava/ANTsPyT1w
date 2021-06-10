## Deploy to AWS Batch

This folder contains the script used to updated an AWS Batch Job Definition, 
deploy_job.sh. The script should be invoked from the top folder of this 
repositiory.

`./deploy/deploy_job.sh 
    <container_name> 
    <cpu_count> 
    <memory_in_gb> 
    <deployment_script>`

`container_name`: The name of the container on the ecr repository

`cpu_count`: the maximum number of cores each instance of this image is allowed to use

`memory_in_gb`: the maximum memory each instance of this image is allows to use

`deployument_script`: the script which defines the processing to be conducted in the container
