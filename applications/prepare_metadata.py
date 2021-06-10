import boto3
import pandas as pd
import datetime


s3 = boto3.client('s3')
bucket = 'eisai-basalforebrainsuperres2'
prefix = 'metadata/'


def get_data(filename, bucket='eisai-basalforebrainsuperres2', prefix='metadata/'):
    s3 = boto3.client('s3')
    x = filename + '.csv'
    full_filename = '/tmp/' + x
    key = prefix + x
    s3.download_file(bucket, key, full_filename)
    df = pd.read_csv(full_filename)
    return df

des = get_data('DESIKANLAB')
mri = get_data('MRIMETA')
mri3 = get_data('MRI3META')
tom = get_data('TOMM40')
meta = get_data('adni_metadata_20210208')

mri_ = pd.concat([mri, mri3], ignore_index=True)



full = pd.merge(mri_, tom, on='RID', how='left',suffixes=("","_drop"))
full = pd.merge(meta, full, on=['PTID', 'EXAMDATE'], how='left', suffixes=("", "_drop"))
full = pd.merge(full, des, on='RID', how='left', suffixes=("", "_drop"))
full = full.drop([x for x in full if x.endswith('_drop')], 1) 

full['Repeat'] = [str(i).split('-')[-1].split('.')[0] for i in full['filename']]
full.to_csv('/tmp/full_metadata_20210208.csv', index=False)
out_key = 'metadata/full_metadata_20210208.csv'
s3.upload_file(
    '/tmp/full_metadata_20210208.csv',
    bucket,
    out_key
)

