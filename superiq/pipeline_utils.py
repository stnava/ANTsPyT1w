import os
import ants
import pandas as pd
import boto3
import json
import sys
import ast

def get_s3_object(bucket, key, local_dir):
    """
    Download an object from an s3 location to specific location locally

    Arguments
    ---------
    bucket : string
        the name of the s3 bucket

    key : string
        the full s3 key of the object

    local_dir : string
        the folder name to download the object to, if folder does not exist one
        is created

    Returns
    -------
    local : string
        the local, relative path to the newly downloaded object

    Example
    -------
    >>> local = get_s3_object("mybucket", "prefix/object_name.txt", "data")
    >>> print(local)
    "data/object_name.txt"
    """
    s3 = boto3.client('s3')
    basename = key.split('/')[-1]
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local = f'{local_dir}/{basename}'
    s3.download_file(
            bucket,
            key,
            local,
    )
    return local


class LoadConfig:
    """
    Creates a config object for referencing the paramerter variables in the passed
    config file.

    Arguments
    ---------
    config : string
        A string that can be parsed into a dict via ast.literal_eval or a
        string representing the file path to the config json file

    Example
    -------
    >>> conf_str = '{"version": 1, "parameters": {"process_name": "my_config"}}'
    >>> config = LoadConfig(conf_str)
    >>> print(config.process_name)
    "my_config"
    """
    def __init__(self, config):
        try:
            # Parse a json string into a python dict (AWS BATCH)
            params = ast.literal_eval(config)
        except ValueError:
            # If using actual json file
            with open(config, 'r') as f:
                data = f.read()
            params = json.loads(data)

        parameters = params['parameters']
        for key in parameters:
            setattr(self, key, parameters[key])

        if params['aws_profile'] != "role":
            os.environ['AWS_PROFILE'] = params['aws_profile']
        print(f"AWS profile set to {params['aws_profile']}")

    def __repr__(self):
        return f"config: {self.__dict__}"


def handle_outputs(input_path, output_bucket, output_prefix, process_name, local_output_dir="outputs", env="prod"):
    """
    Uploads all files in the outputs dir to the appropriate location on s3

    Arguments
    ---------
    input_path : string
        the original pipeline image path or key

    output_bucket : string
        the name of the bucket to upload data to

    output_prefix : string
        the name of the prefix to prepend to the derived s3 prefix

    process_name : string
        the name of the process run suffixed to the derived s3 prefix

    dev  : boolean
        on/off switch to disable uploads to s3 during testing (True=No Upload)

    Example
    -------
    >>> handle_outputs(
            "data/ADNI-002_S_0413-20060224-T1w-000.nii.gz",
            "my_bucket",
            "output_data/",
            "my_process",
            dev=True
        )
    *print*
    "data/ADNI-002_S_0413-20060224-T1w-000.nii.gz ->
        output_data/ADNI/002_S_0413/20060224/T1w/000/my_process/ADNI-002_S_0413-20060224-T1w-000-my_process-new_data.nii.gz"
    """
    outputs = [i for i in os.listdir(local_output_dir)]
    path, basename = derive_s3_path(input_path)
    prefix = output_prefix + path + process_name + '/'
    s3 = boto3.client('s3')
    for output in outputs:
        filename = output.split('/')[-1]
        outpath = local_output_dir + "/" + output
        obj_name = basename + '-' + process_name + '-' + filename
        obj_path = prefix + obj_name
        print(f"{outpath} -> {obj_path}")
        if env == "prod":
            s3.upload_file(
                    outpath,
                    output_bucket,
                    obj_path,
            )


def get_pipeline_data(filename, initial_image_key, bucket, prefix, local_dir="data"):
    """
    Retrieve data ending with a certain name from s3 based on the original input image

    Arguments
    ---------
    filename : string
        the filename ending to match

    initial_image_key :  string
        the file name of the original input image to the pipeline

    bucket : string
        the bucket name where the object to match resides

    prefix : string
        the prefix in the bucekt under with the object to match resides

    local_dir : string
        the folder to place the output

    Return
    ------
    local : string
        the local path of the downloaded object

    Example
    -------
    >>> data = get_pipeline_data(
            "bxtreg_n3.nii.gz",
            "data/ADNI-002_S_0413-20060224-T1w-000.nii.gz",
            "my_bucket",
            "output_data/",
        )
    >>> print(data)
    "data/ADNI-002_S_0413-20060224-T1w-000-brain_ext-bxtreg_n3.nii.gz"
    """

    path, _ = derive_s3_path(initial_image_key)
    key_list = list_images(bucket, prefix + path)
    key = [i for i in key_list if i.endswith(filename)]
    if len(key) != 1:
        raise ValueError(f'{len(key)} objects were found with that suffix')
    else:
        key = key[0]
        local = get_s3_object(bucket, key, local_dir)
        return local


def derive_s3_path(image_path):
    """
    Given a path or s3 key, derives the expected s3 prefix based on the base filename

    Arguments
    ---------
    image_path : string
        the s3 key or file path to an nii.gz file

    Return
    ------

    path, basename : tuple (string, string)
        the path of the image as it would appear on s3 and the basename of the image without
        the extension

    Example
    -------
    >>> path, basename = derive_s3_path("data/ADNI-002_S_0413-20060224-T1w-000.nii.gz")
    >>> print(path)
    "ADNI/002_S_0413/20060224/T1w/000/brain_ext/"
    >>> print(basename)
    "ADNI-002_S_0413-20060224-T1w-000",
    """
    basename = image_path.split('/')[-1].replace('.nii.gz', '')
    loc = basename.split('-')
    path = '/'.join(loc) + '/'
    return path, basename


def list_images(bucket, prefix):
    """
    Helper function for listing all file objects with an extension in s3 under
    a bucket and prefix

    Arguments
    ---------
    bucket : string
        the bucket to search for objects

    prefix : string
        the prefix to search for objects

    Return
    ------
    images : list (strings)
        the s3 keys of the objects with an extension in the bucket and prefix

    Example
    -------
    >>> keys = list_images("my_bucket", "my_prefix/")
    >>> print(keys)
    ["my_prefix/object_A.json", "my_prefix/subfolder/object_B.json"]
    """
    s3 = boto3.client('s3')
    items = []
    kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }
    while True:
        objects = s3.list_objects_v2(**kwargs)
        try:
            for obj in objects['Contents']:
                key =  obj['Key']
                items.append(key)
        except KeyError:
            raise KeyError(f"No keys found under prefix {prefix}")
        try:
            kwargs['ContinuationToken'] = objects['NextContinuationToken']
        except KeyError:
            break
    images = [i for i in items if "." in i]
    return images


def get_label_geo(
        labeled_image, # The ants image to be labeled
        initial_image, # The unlabeled image, used in the label stats
        process, # Name of the process
        input_key, # Replace
        label_map_params=None, # Ignore for now
        resolution='OR',
        direction=None):
    print('Starting Label Geometry Measures')
    lgms = ants.label_geometry_measures(
            labeled_image,
    )
    if label_map_params is not None:
        label_map_local = get_input_image(label_map_params['bucket'], label_map_params['key'])
        label_map = pd.read_csv(label_map_local)
        label_map.columns = ['LabelNumber', 'LabelName']
        label_map.set_index('LabelNumber', inplace=True)
        label_map_dict = label_map.to_dict('index')

    new_rows = []
    for i,r in lgms.iterrows():
        label = int(r['Label'])
        if label_map_params is not None:
            label = label_map_dict[label]['LabelName']
            clean_label = label.replace(' ', '_')
        else:
            clean_label = label
        fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
        select_data = r[fields]
        values = select_data.values
        field_values = zip(fields, values)

        for f in field_values:
            new_df = {}
            new_df['Measure'] = f[0]
            new_df['Value'] = f[1]
            new_df['Process'] = process
            new_df['Resolution'] = resolution
            if direction is not None:
                new_df['Side'] = direction
            else:
                new_df['Side'] = 'full'
            new_df['Label'] = clean_label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)
    label_stats = ants.label_stats(initial_image, labeled_image)
    label_stats = label_stats[label_stats['Mass']>0]

    for i,r in label_stats.iterrows():
        label = int(r['LabelValue'])
        if label_map_params is not None:
            label = label_map_dict[label]['LabelName']
            clean_label = label.replace(' ', '_')
        else:
            clean_label = label
        fields = ['Mean']
        select_data = r[fields]
        values = select_data.values
        field_values = zip(fields, values)
        for f in field_values:
            new_df = {}
            new_df['Measure'] = 'MeanIntensity'
            new_df['Value'] = f[1]
            new_df['Process'] = process
            new_df['Resolution'] = resolution
            if direction is not None:
                new_df['Side'] = direction
            else:
                new_df['Side'] = 'full'
            new_df['Label'] = clean_label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)
    label_data = pd.concat(new_rows)

    s3_path, _ = derive_s3_path(input_key)
    split = s3_path.split('/')

    label_data['Study'] = split[0]
    label_data['Subject'] = split[1]
    label_data['Date'] = split[2]
    label_data['Modality'] =  split[3]
    label_data['Repeat'] = split[4]
    full = label_data
    if direction is None:
        side = "full"
    else:
        side = direction
    output_name = f'outputs/{resolution}-{side}-lgm.csv'
    full.to_csv(output_name, index=False)


def plot_output(img, output_path, overlay=None):
    if overlay is None:
        plot = ants.plot_ortho(
                ants.crop_image(img),
                flat=True,
                filename=output_path,
        )
    else:
        plot = ants.plot_ortho(
                ants.crop_image(img, overlay),
                overlay=ants.crop_image(overlay, overlay),
                flat=True,
                filename=output_path,
        )

