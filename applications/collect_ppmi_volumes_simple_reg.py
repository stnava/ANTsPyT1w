from superiq import VolumeData
from superiq.pipeline_utils import *
import boto3
import pandas as pd
from datetime import datetime

def collect_brain_age():
	bucket = "mjff-ppmi"
	version = "simple-v2"
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
	#metadata_key = "volume_measures/data_w_metadata_v01.csv"
	version = "simple-v2"
	prefix = f"superres-pipeline-{version}/"
	stack_filename = f'ppmi_stacked_volumes_{version}.csv'
	pivoted_filename = f'ppmi_pivoted_volumes_{version}.csv'
	#merge_filename = f"dkt_with_metdata_{version}.csv"
	upload_prefix = "volume_measures/"
	vd = VolumeData(bucket, prefix, upload_prefix, cache=False)
	local_stack = vd.stack_volumes(stack_filename)
	local_pivot = vd.pivot_data(local_stack, pivoted_filename)
	local_pivot_df = pd.read_csv(local_pivot)
	local_pivot_df = local_pivot_df
	ba = collect_brain_age()
	local_pivot_df['join_date'] = [str(i)[:6] for i in local_pivot_df['Date']]
	print(local_pivot_df.shape)
	local_pivot_df = pd.merge(local_pivot_df, ba, on='Repeat')
	print(local_pivot_df.shape)
	s3 = boto3.client('s3')
	local_pivot_df.to_csv('local_pivot.csv')
	s3.upload_file(
		'local_pivot.csv',
		bucket,
		upload_prefix + "simple_reg_sr_ppmi_volumes.csv"
	)
	metadata = 'metadata/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv'
	prodro = 'metadata/PPMI_Prodromal_Cohort_BL_to_Year_1_Dataset_Apr2020.csv'

	metadata_path = 'ppmi_metadata.csv'
	prodro_path = 'ppmi_prodro.csv'
	s3.download_file(bucket, metadata, metadata_path)
	s3.download_file(bucket, prodro, prodro_path)

	metadata_df = pd.read_csv(metadata_path)
	prodro_df = pd.read_csv(prodro_path)
	stack = pd.concat([metadata_df, prodro_df])
	stack['join_date'] = [datetime.strptime(i, '%b%Y') for i in stack['visit_date']]
	stack['join_date'] = [i.strftime('%Y%m') for i in stack['join_date']]
	stack['PATNO'] = [str(i) for i in stack['PATNO']]
	# Join on Subject
	merged = pd.merge(
		stack,
		local_pivot_df,
		right_on=['Subject', 'join_date'],
		left_on=['PATNO', 'join_date'],
		how='outer'
	)
	merged_path = "full_" + "simple_reg_sr_ppmi_volumes.csv"
	merged.to_csv(merged_path, index=False)
	s3.upload_file(merged_path, bucket, upload_prefix +  merged_path)

	print(merged[merged['PATNO'].isna()].shape)
