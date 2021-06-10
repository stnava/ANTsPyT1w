from superiq.pipeline_utils import *
import pandas
import os
import boto3


bucket = "eisai-basalforebrainsuperres2"
prefix = f"superres-pipeline-validation/deep_dkt/"
stack_filename = f'stacked_dice_scores.csv'
output_name = "deep_dkt_dice_scores_all.csv"
s3 = boto3.client('s3')

def get_files(k):
    bucket = "eisai-basalforebrainsuperres2" # Param
    path = get_s3_object(bucket, k, "tmp")
    return path

def main():
    keys = list_images(bucket, prefix)
    keys = [i for i in keys if ".csv" in i]
    keys = [i for i in keys if "_all" not in i]
    dfs = [pd.read_csv(get_files(i)) for i in keys]
    stacked = pd.concat(dfs)
    stacked.to_csv(stack_filename, index=False)
    s3.upload_file(stack_filename, bucket, prefix + output_name)

if __name__ == "__main__":
    main()
