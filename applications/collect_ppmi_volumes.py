from superiq import VolumeData
from superiq.pipeline_utils import *
import boto3
import pandas as pd


if __name__ == "__main__":
      bucket = "mjff-ppmi"
      #metadata_key = "volume_measures/data_w_metadata_v01.csv"
      version = "ljlf-right"
      prefix = f"superres-pipeline-{version}/"
      stack_filename = f'ppmi_stacked_volumes_{version}.csv'
      pivoted_filename = f'ppmi_pivoted_volumes_{version}.csv'
      #merge_filename = f"dkt_with_metdata_{version}.csv"
      upload_prefix = "volume_measures/"
      vd = VolumeData(bucket, prefix, upload_prefix, cache=False)
      local_stack = vd.stack_volumes(stack_filename)
      local_pivot_r = vd.pivot_data(local_stack, pivoted_filename)
      local_pivot_r = pd.read_csv(local_pivot_r)


      version = "ljlf-left"
      prefix = f"superres-pipeline-{version}/"
      stack_filename = f'ppmi_stacked_volumes_{version}.csv'
      pivoted_filename = f'ppmi_pivoted_volumes_{version}.csv'
      #merge_filename = f"dkt_with_metdata_{version}.csv"
      upload_prefix = "volume_measures/"
      vd = VolumeData(bucket, prefix, upload_prefix, cache=False)
      local_stack = vd.stack_volumes(stack_filename)
      local_pivot_l = vd.pivot_data(local_stack, pivoted_filename)
      local_pivot_l = pd.read_csv(local_pivot_l)
      joined = pd.merge(local_pivot_l, local_pivot_r, on="Repeat", how='outer', suffixes=('', '_x'))
      drops = [i for i in joined.columns if "_x" in i]
      joined = joined.drop(drops, axis='columns')
      joined.to_csv('ppmi_volumes.csv')
      s3 = boto3.client('s3')
      s3.upload_file('ppmi_volumes.csv', bucket, "ppmi_volumes.csv")
