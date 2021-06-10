# this script assumes the image have been brain extracted
import os.path
from os import path

# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import ljlf_parcellation_one_template
from superiq import list_to_string
from pipeline_utils import *


# user definitions here
#tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
#sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
#model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
#tfn = tdir + "CIT168_T1w_700um.nii.gz"
#tfnl = tdir + "det_atlas_25.nii.gz"
#infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
#wlab = range(1,16) # mid-brain regions in CIT168

# config handling
output_filename = "outputs/CITI68"
config = LoadConfig('configs/cit168_one_template_example.json')

tfn = get_s3_object(config.template_bucket, config.template_key, "data")
tfnl = get_s3_object(config.template_bucket, config.template_label_key, "data")
infn = get_pipeline_data(
        "brain_ext-bxtreg_n3.nii.gz",
        config.input_value,
        config.pipeline_bucket,
        config.pipeline_prefix,
)
model_file_name = get_s3_object(config.model_bucket, config.model_key, "models")
seg_params = config.seg_params
if seg_params['wlab']['range']:
    wlab = range(seg_params['wlab']['values'][0],seg_params['wlab']['values'][1])
else:
    wlab = seg_params['wlab']['values']

# input data
imgIn = ants.image_read( infn )
template = ants.image_read(tfn)
templateL = ants.image_read(tfnl)
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

# expected output data
output_filename_seg = output_filename + "_ORseg.nii.gz"
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"


# just run - note : DANGER - we might skip this if it is already there so take
# care to run with a clean output directory or new output prefix
#if ( not 'dktpar' in locals() ) & ( not os.path.isfile(output_filename_seg) ):
if True:
    if not 'reg' in locals():
        print("SyN begin")
        reg = ants.registration( imgIn, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( imgIn, templateL,
          forward_transforms, interpolator="nearestNeighbor" )
        print("SyN done")
    locseg = ljlf_parcellation_one_template(
        imgIn,
        segmentation_numbers=wlab,
        forward_transforms=forward_transforms,
        template=template,
        templateLabels=templateL,
        templateRepeats=seg_params['template_repeats'],
        submask_dilation=seg_params['submask_dilation'],  # a parameter that should be explored
        searcher=seg_params['searcher'],  # double this for SR
        radder=seg_params['radder'],  # double this for SR
        reg_iterations=seg_params['reg_iterations'], # fast test
        output_prefix=output_filename,
        verbose=seg_params['verbose'],
    )
    ants.image_write( locseg['segmentation'], output_filename_seg )
    get_label_geo(
            locseg['segmentation'],
            imgIn,
            config.process_name,
            config.input_value,
            resolution='OR',
    )
    plot_output(
            imgIn,
            'outputs/OR_ortho_plot.png',
            locseg['segmentation'],
    )

localseg = ants.image_read( output_filename_seg )

# NOTE: the code below is SR specific and should only be run in that is requested
if hasattr(config, "sr_params"):
    sr_params = config.sr_params
    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = localseg,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = wlab,
        dilation_amount = sr_params['dilation_amount'],
        max_lab_plus_one = True,
        verbose = sr_params['verbose']
    )
    get_label_geo(
            srseg['super_resolution_segmentation'],
            srseg['super_resolution'],
            config.process_name,
            config.input_value,
            resolution='SR',
    )
    plot_output(
            srseg['super_resolution'],
            'outputs/SR_ortho_plot.png',
            srseg['super_resolution_segmentation'],
    )

    ants.image_write( srseg['super_resolution'], output_filename_sr )
    ants.image_write(srseg['super_resolution_segmentation'], output_filename_sr_seg )

handle_outputs(
        config.input_value,
        config.output_bucket,
        config.output_prefix,
        config.process_name,
        dev=False, # False = outputs uploaded to s3 location
)
