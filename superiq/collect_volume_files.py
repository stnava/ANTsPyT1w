import ants
from superiq.pipeline_utils import *
import pandas
import os
import boto3
import multiprocessing as mp
from datetime import datetime

class VolumeData:

    def __init__(self, bucket, prefix, filter_suffixes, upload_prefix):
        self.bucket = bucket
        self.prefix = prefix
        self.filter_suffixes = filter_suffixes
        self.upload_prefix = upload_prefix

    def stack_volumes(self, stack_filename):
        print("====> Stacking volumes")
        keys = self._filter_keys(self.filter_suffixes)
        with mp.Pool() as p:
            dfs = p.map(self._get_files, keys)
        stacked = pd.concat(dfs)
        stack_filename_key = f"s3://{self.bucket}/{stack_filename}"
        stacked.to_csv(stack_filename_key, index=False)
        #key = self.upload_file(stack_filename)
        return stack_filename_key

    def _filter_keys(self, filter_suffix):
        print("====> Getting keys")
        keys = list_images(self.bucket, self.prefix)
        filtered_keys = []
        for fil in filter_suffix:
            print(fil)
            keys2 = [i for i in keys if i.endswith(fil)]
            print(len(keys2))
            for k in keys2:
                filtered_keys.append(k)
        return filtered_keys

    def _get_files(self, k):
        #path = get_s3_object(bucket, k, "tmp/")
        df = pd.read_csv(f"s3://{self.bucket}/{k}")
        fields = ["Label", 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
        df = df[fields]
        new_rows = []
        for i,r in df.iterrows():
            label = int(r['Label'])
            fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
            select_data = r[fields]
            values = select_data.values
            field_values = zip(fields, values)
            for f in field_values:
                new_df = {}
                new_df['Measure'] = f[0]
                new_df['Value'] = f[1]
                new_df['Label'] = label
                new_df = pd.DataFrame(new_df, index=[0])
                new_rows.append(new_df)

        df = pd.concat(new_rows)
        filename = k.split('/')[-1]
        split = filename.split('-')
        name = split[6].split('.',1)[0]
        name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process"]
        zip_list = zip(name_list, split)
        for i in zip_list:
            df[i[0]] = i[1]
        df['OriginalOutput'] = "-".join(split[:5]) + ".nii.gz"
        if "SR" in k:
            df['Resolution'] = "SR"
        else:
            df['Resolution'] = "OR"
        df['Name'] = name
        return df

    def pivot_data(self, stack_filename_key, pivot_filename):
        print("====> Pivoting Data")
        df = pd.read_csv(stack_filename_key)
        #df['Name'] = [i.split('.')[0] for i in df['Name']]
        pivoted = df.pivot(
            index=['Project','Subject','Date', 'Modality', 'Repeat',"OriginalOutput"],
            columns=['Measure', 'Label',"Resolution",'Process',"Name"])

        columns = []
        for c in pivoted.columns:
            cols = [str(i) for i in c]
            column_name = '-'.join(cols[1:])
            columns.append(column_name)

        pivoted.columns = columns
        pivoted.reset_index(inplace=True)
        final_csv = pivoted

        pivot_filename_key = f"s3://{self.bucket}/{pivot_filename}"
        final_csv['Repeat'] = [str(i).zfill(3) for i in final_csv['Repeat']]
        final_csv.to_csv(pivot_filename_key, index=False)
        #self.upload_file(pivot_filename)
        return pivot_filename_key

    #def upload_file(self, filename):
    #    s3 = boto3.client('s3')
    #    key = self.upload_prefix + filename
    #    s3.upload_file(
    #        filename,
    #        self.bucket,
    #        key
    #    )
    #    return key



    def merge_data_with_metadata(self,
                                 pivoted_filename,
                                 metadata_key,
                                 merge_filename,
                                 on=['filename','OriginalOutput'],
    ):
        data = pd.read_csv(pivoted_filename)
        meta = get_s3_object(self.bucket, metadata_key, "tmp")
        metadf = pd.read_csv(meta)
        os.remove(meta)
        merge = pd.merge(
            metadf,
            data,
            how="outer",
            left_on=on[0],
            right_on=on[1],
            suffixes=("","_x")
        )
        duplicate_columns = [i for i in merge.columns if i.endswith('_x')]
        merge.drop(duplicate_columns, inplace=True, axis=1)
        merge.to_csv(merge_filename, index=False)
        self.upload_file(merge_filename)

def fix_dt(dt_col):
    dt_col = [str(i) for i in dt_col]
    fixed_dt = [i.split('/')[-1]+i.split('/')[0] for i in dt_col]
    return(fixed_dt)

if __name__ == "__main__":
    direct = False
    if direct:
        bucket = "mjff-ppmi"
        #metadata_key = "volume_measures/data_w_metadata_v01.csv"
        version = "mjff"
        prefix = "superres-pipeline-mjff/"
        upload_prefix = "volume_measures/"
        stack_filename = upload_prefix + f'stacked_volumes_{version}.csv'
        pivoted_filename = upload_prefix + f'pivoted_volumes_{version}.csv'
        #merge_filename = f"dkt_with_metdata_{version}.csv"
        filter_suffixes = ['OR_seg.csv', "SR_ljflseg.csv", "SR_seg.csv", "SR_regseg.csv"]
        vd = VolumeData(bucket, prefix,filter_suffixes, upload_prefix)
        local_stack = vd.stack_volumes(stack_filename)
        local_pivot = vd.pivot_data(local_stack, pivoted_filename)
        #vd.merge_data_with_metadata(local_pivot, metadata_key, merge_filename)

    bxt = False
    if bxt:
        bucket = "mjff-ppmi"
        #metadata_key = "volume_measures/data_w_metadata_v01.csv"
        version = "bxt"
        prefix = "t1_brain_extraction_v2/"
        upload_prefix = "volume_measures/"
        stack_filename = upload_prefix + f'stacked_volumes_{version}.csv'
        pivoted_filename = upload_prefix + f'pivoted_volumes_{version}.csv'
        #merge_filename = f"dkt_with_metdata_{version}.csv"
        filter_suffixes = ['brainvol.csv']
        bxt = VolumeData(bucket, prefix,filter_suffixes, upload_prefix)
        bxt_stack = bxt.stack_volumes(stack_filename)
        bxt_pivot = bxt.pivot_data(bxt_stack, pivoted_filename)
        #vd.merge_data_with_metadata(local_pivot, metadata_key, merge_filename)


    local_pivot = "s3://mjff-ppmi/volume_measures/pivoted_volumes_mjff.csv"
    bxt_pivot = "s3://mjff-ppmi/volume_measures/pivoted_volumes_bxt.csv"
    output = "s3://mjff-ppmi/volume_measures/ppmi_all_volumes.csv"
    direct_df = pd.read_csv(local_pivot)
    bxt_df = pd.read_csv(bxt_pivot)
    print(direct_df.shape)
    print(bxt_df.shape)
    volumes = pd.merge(bxt_df, direct_df, on="Repeat", suffixes=("", "_x"))
    print(volumes.shape)
    duplicate_columns = [i for i in volumes.columns if i.endswith('_x')]
    volumes.drop(duplicate_columns, inplace=True, axis=1)
    print(volumes.shape)

    metadata_df = pd.read_csv("s3://mjff-ppmi/metadata/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv")
    print("Metadata")
    print(metadata_df.shape)
    prodro_df = pd.read_csv("s3://mjff-ppmi/metadata/PPMI_Prodromal_Cohort_BL_to_Year_1_Dataset_Apr2020.csv")
    print(prodro_df.shape)
    stack = pd.concat([metadata_df, prodro_df])
    print(stack.shape)
    mri_join_map = 's3://mjff-ppmi/metadata/mri_join_map.csv'
    mri_join_map = pd.read_csv(mri_join_map)
    mri_join_map['mridt'] = fix_dt(mri_join_map['mridt'])
    mri_join_map['infodt'] = fix_dt(mri_join_map['infodt'])
    mri_join_map['patno'] = [str(i) for i in mri_join_map['patno']]


    stack['join_date'] = [datetime.strptime(i, '%b%Y') for i in stack['visit_date']]
    stack['join_date'] = [str(i.strftime('%Y%m')) for i in stack['join_date']]
    volumes['mri_date'] = [str(i)[:6] for i in volumes['Date']]
    volumes['Subject'] = [str(i) for i in volumes['Subject']]
    stack['PATNO'] = [str(i) for i in stack['PATNO']]
    # Join on Subject
    mridt_added = pd.merge(
            stack,
            mri_join_map,
            left_on=['PATNO', 'join_date'],
            right_on=['patno', 'infodt'],
            how='left'
    )
    print('MRI dates')
    print(mridt_added.shape)
    merged = pd.merge(
            mridt_added,
            volumes,
            left_on=['PATNO', 'mridt'],
            right_on=['Subject', 'mri_date'],
            how='outer'
    )
    print(merged.shape)
    output_uri = "s3://mjff-ppmi/volume_measures/direct_regseg_bxt_volumes.csv"
    merged.to_csv(output_uri, index=False)
