from superiq import VolumeData
from superiq.pipeline_utils import *
import boto3
import pandas as pd
from datetime import datetime

def collect_brain_age(bucket, version):
	prefix = f"superres-pipeline-{version}/"
	objects = list_images(bucket, prefix)
	brain_age = [i for i in objects if i.endswith('brain_age.csv')]
	dfs = []
	for i in brain_age:
		ba = get_s3_object(bucket, i, '/tmp')
		filename = ba.split('/')[-1]
		splits = filename.split('-')
		ba_df = pd.read_csv(ba)
		ba_df['Repeat'] = splits[4]
		dfs.append(ba_df)
	dfs = pd.concat(dfs)
	return dfs

if __name__ == "__main__":
	bucket = "mjff-ppmi"
	version = "mjff"
	prefix = f"superres-pipeline-{version}/"
	stack_filename = f'ppmi_stacked_volumes_{version}.csv'
	pivoted_filename = f'ppmi_pivoted_volumes_{version}.csv'
	upload_prefix = "volume_measures/"
	filter_suffixes = ['OR_seg.csv', 'SR_jlfseg.csv','SR_ljflseg.csv', 'SR_seg.csv', 'SR_regseg.csv']
	vd = VolumeData(bucket, prefix, filter_suffixes, upload_prefix)
	local_stack = vd.stack_volumes(stack_filename)
	local_pivot = vd.pivot_data(local_stack, pivoted_filename)
	local_pivot_df = pd.read_csv(local_pivot)
	local_pivot_df = local_pivot_df
	ba = collect_brain_age(bucket, version)
	local_pivot_df['join_date'] = [str(i)[:6] for i in local_pivot_df['Date']]
	local_pivot_df = pd.merge(local_pivot_df, ba, on='Repeat')
	s3 = boto3.client('s3')
	local_pivot_df.to_csv('local_pivot.csv')
	s3.upload_file('local_pivot.csv', bucket, f"volume_measures/direct_reg_seg_ppmi_volumes-{version}.csv")
	metadata = False
	if metadata:
		metadata_bucket = 'mjff-ppmi'
		metadata_key = 's3://ppmi-metadata/PPMIFullMetadata.csv'
		metadata_df = pd.read_csv(metadata_key)
		merged = pd.merge(
			metadata_df,
			local_pivot_df,
			right_on=['Subject', 'join_date'],
			left_on=['PATNO', 'join_date'],
			how='outer'
		)
		merged_path = "full_" + "simple_reg_sr_ppmi_volumes.csv"
		merged.to_csv(merged_path, index=False)
		s3.upload_file(merged_path, bucket, merged_path)
