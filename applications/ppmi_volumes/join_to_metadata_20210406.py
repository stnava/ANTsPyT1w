import pandas as pd
import boto3

name = "-20210406"
metadata = "s3://ppmi-metadata/derived_tables/demog_ppmi_built_07042021.csv"
cst = True
cst = f's3://mjff-ppmi/volume_measures/direct_reg_seg_ppmi_volumes-mjff{name}-cst.csv'
dir_reg_seg = f's3://mjff-ppmi/volume_measures/direct_reg_seg_ppmi_volumes-mjff.csv'#{name}.csv'
randbasis = 's3://mjff-ppmi/superres-pipeline-mjff-randbasis/fullprojs.csv'
randbasis_df = pd.read_csv(randbasis)
randbasis_df.drop(['Subject.ID','Date'], inplace=True, axis=1)

metadata_df = pd.read_csv(metadata)
drs = pd.read_csv(dir_reg_seg)

print(metadata_df.shape)
drs['Image.ID'] = [int(i.replace('I', '')) for i  in drs['Repeat']]
join = pd.merge(metadata_df, drs, on='Image.ID', how='left')

randbasis_df['Image.ID'] = [int(i.replace('I', '')) for i  in randbasis_df['Image.ID']]
join = pd.merge(join, randbasis_df, on='Image.ID', how='left')

output = f's3://mjff-ppmi/volume_measures/VolumesDemogPPMI{name}.csv'
join.to_csv(output)
print(join.shape)
if cst:
	cst_df = pd.read_csv(cst)
	cst_df['Image.ID'] = [int(i.replace('I', '')) for i  in cst_df['Repeat']]
	join = pd.merge(join, cst_df, on='Image.ID', how='left')
	join.to_csv(output, index=False)
	print(join.shape)
