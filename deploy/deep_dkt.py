import os
threads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import antspynet
import math
import ants
import sys
#from superiq.pipeline_utils import *
from superiq import deep_dkt, super_resolution_segmentation_per_label
import ia_batch_utils as batch
import pandas as pd


def main(input_config):
    config = batch.LoadConfig(input_config)
    c = config
    if config.environment == 'prod':

        input_image = batch.get_s3_object(
                config.input_bucket,
                config.input_value,
                "data"
        )

        input_image = ants.image_read(input_image)
    elif config.environment == 'val':
        input_path = batch.get_s3_object(config.input_bucket, config.input_value, 'data')
        input_image = ants.image_read(input_path)
        image_base_name = input_path.split('/')[-1].split('.')[0]
        folder_name =  image_base_name.replace('_t1brain', "") + '/'
        image_label_name = \
            config.label_prefix + folder_name +  image_base_name + '_labels.nii.gz'
        print(image_label_name)
        image_labels_path = batch.get_s3_object(
            config.label_bucket,
            image_label_name,
            "data",
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

    wlab = config.wlab

    template = antspynet.get_antsxnet_data("biobank")
    template = ants.image_read(template)
    template =  template * antspynet.brain_extraction(template, 't1').iMath("Normalize")

    sr_model = batch.get_s3_object(config.model_bucket, config.model_key, "data")
    mdl = tf.keras.models.load_model(sr_model)

    output_path = config.output_file_prefix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_params={
        "target_image": input_image,
        "segmentation_numbers": wlab,
        "template": template,
        "sr_model": sr_model,
        "sr_params": config.sr_params,
        "output_path": output_path,
    }

    output = deep_dkt(**input_params)

    label_or = output['labels_or']
    label_sr = output['labels_sr']

    labels_or = pd.read_csv(label_or)
    labels_sr = pd.read_csv(label_sr)

    vals = {
        "OR": labels_or,
        "SR": labels_sr,
    }

    for key, value in vals.items():
        split = config.input_value.split('/')[-1].split('-')
        rec = {}
        rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
        rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
        rec['batchid'] = c.batch_id
        rec['project'] = split[0]
        rec['subject'] = split[1]
        rec['date'] = split[2]
        rec['modality'] = split[3]
        rec['repeat'] = split[4]
        rec['process'] = 'deep_dkt'
        rec['name'] = "deep_dkt"
        rec['version'] = config..version
        rec['extension'] = ".nii.gz"
        rec['resolution'] = key
        df = value[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]
        volumes = df.to_dict('records')
        for r in volumes:
            label = r['Label']
            r.pop("Label", None)
            for k, v in r.items():
                rec['data'] = {}
                rec['data']['label'] = label
                rec['data']['key'] = k
                rec['data']['value'] = v
                print(rec)
                batch.write_to_dynamo(rec)




    #pivot_csv(labels_or, config, "OR")
    #pivot_csv(labels_sr, config, "SR")

    if config.environment == 'prod':
        batch.handle_outputs(
            config.output_bucket,
            config.output_prefix,
            config.input_value,
            config.process_name,
            config.verions
        )

    elif config.environment == 'val':
        native_seg = output['nativeSeg']
        nativeGroundTruth = ants.image_read(image_labels_path)

        label_one_comp_set = config.label_one_labelset # [17, 53]
        label_one_gt = ants.threshold_image(
            nativeGroundTruth,
            label_one_comp_set[0],
            label_one_comp_set[0],
        )
        label_one = ants.threshold_image(
            native_seg,
            label_one_comp_set[1],
            label_one_comp_set[1],
        )
        dice_one = ants.label_overlap_measures(label_one_gt, label_one)
        dice_one = dice_one['MeanOverlap'][1]
        print(dice_one)

        label_two_comp_set = config.label_two_labelset # [1006, 2006]
        label_two_gt = ants.threshold_image(
            nativeGroundTruth,
            label_two_comp_set[0],
            label_two_comp_set[0],
        )
        label_two = ants.threshold_image(
            native_seg,
            label_two_comp_set[1],
            label_two_comp_set[1],
        )
        dice_two = ants.label_overlap_measures(label_two_gt, label_two)
        dice_two = dice_two['MeanOverlap'][1]
        print(dice_two)

        label_three_comp_set = config.label_three_labelset # [1006, 2006]
        label_three_gt = ants.threshold_image(
            nativeGroundTruth,
            label_three_comp_set[0],
            label_three_comp_set[0],
        )
        label_three = ants.threshold_image(
            native_seg,
            label_three_comp_set[1],
            label_three_comp_set[1],
        )
        dice_three = ants.label_overlap_measures(label_three_gt, label_three)
        dice_three = dice_three['MeanOverlap'][1]
        print(dice_three)

        brainName = []
        col1 = []
        col2 = []
        col3 = []

        brainName.append(image_base_name)
        col1.append(dice_one)
        col2.append(dice_two)
        col3.append(dice_three)

        dict = {
                'name': brainName,
                'hippocampus': col1,
                'entorhinal': col2,
                'parahippocampal': col3,
        }
        df = pd.DataFrame(dict)
        path = f"{image_base_name}_dice_scores.csv"
        df.to_csv(output_path + path, index=False)
        s3 = boto3.client('s3')
        s3.upload_file(
            output_path + path,
            config.output_bucket,
            config.output_prefix + path,
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

def pivot_csv(df, config, resolution):
    #df = pd.read_csv(f"s3://{config.input_bucket}/{k}")
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
    filename = config.input_value.split('/')[-1]
    split = filename.split('-')[:5]
    name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process", "Resolution"]
    split.append('deepdkt')
    split.append(resolution)
    zip_list = zip(name_list, split)
    for i in zip_list:
        df[i[0]] = i[1]
    df['OriginalOutput'] = "_".join(split[:5]) + ".nii.gz"
    df['Name'] = filename
    #return df
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
    labels = '_'.join([str(i) for i in config.wlab])
    output_name = f"Labels_{labels}_{resolution}.csv"
    final_csv['Repeat'] = [str(i).zfill(3) for i in final_csv['Repeat']]
    final_csv.to_csv(f'outputs/{output_name}', index=False)


if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
