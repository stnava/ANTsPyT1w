import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import os.path
from os import path
import glob as glob


import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import check_for_labels_in_image
from pipeline_utils import *
from superiq import list_to_string


# user definitions here
#tdir = "/Users/stnava/code/super_resolution_pipelines/data/OASIS30/"
#brains = glob.glob(tdir+"Brains/*")
#brains.sort()
#brains = brains[0:8] # shorten this for the test application
#brainsSeg = glob.glob(tdir+"Segmentations/*")
#brainsSeg.sort()
#brainsSeg = brainsSeg[0:8] # shorten this for this test application
#templatefilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template.nii.gz"
#templatesegfilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template_dkt_labels.nii.gz"
#sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
#model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
#infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
#output_filename = "outputs3/ADNI_BF"
#wlab = [ 75, 76 ] # basal forebrain in OASIS

# config handling
output_filename = "outputs/basalforebrain"
config = LoadConfig('configs/basal_forebrain_multiatlas_example_SR_first.json')

templatefilename = get_s3_object(config.template_bucket, config.template_key, "data")
templatesegfilename = get_s3_object(config.template_bucket, config.template_label_key, "data")
infn = get_pipeline_data(
        "brain_ext-bxtreg_n3.nii.gz",
        config.input_value,
        config.pipeline_bucket,
        config.pipeline_prefix,
)
model_file_name = get_s3_object(config.model_bucket, config.model_key, "models")
atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
brains.sort()
brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
brainsSeg.sort()
print(brains)
print(brainsSeg)

sr_params = config.sr_params

seg_params = config.seg_params
if seg_params['wlab']['range']:
    wlab = range(seg_params['wlab']['values'][0],seg_params['wlab']['values'][1])
else:
    wlab = seg_params['wlab']['values']

# input data
imgIn = ants.image_read( infn )
template = ants.image_read( templatefilename )
templateL = ants.image_read( templatesegfilename )
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

havelabels = check_for_labels_in_image( wlab, templateL )

if not havelabels:
    raise Exception("Label missing from the template")

# expected output data
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"


# first, run registration - then do SR in the local region
if not 'reg' in locals():
    reg = ants.registration( imgIn, template, 'SyN' )
    forward_transforms = reg['fwdtransforms']
    initlab0 = ants.apply_transforms( imgIn, templateL,
          forward_transforms, interpolator="genericLabel" )

srseg = super_resolution_segmentation_per_label(
    imgIn = imgIn,
    segmentation = initlab0,
    upFactor = sr_params['upFactor'],
    sr_model = mdl,
    segmentation_numbers = wlab,
    dilation_amount = sr_params['dilation_amount'],
    verbose = sr_params['verbose']
)

# write
initlab0 = ants.apply_transforms( srseg['super_resolution'], templateL,
    forward_transforms, interpolator="genericLabel" )
ants.image_write( srseg['super_resolution'] , output_filename_sr )
ants.image_write( initlab0 , output_filename_sr_seg_init )

locseg = ljlf_parcellation(
        srseg['super_resolution'],
        segmentation_numbers=wlab,
        forward_transforms=forward_transforms,
        template=template,
        templateLabels=templateL,
        library_intensity = brains,
        library_segmentation = brainsSeg,
        submask_dilation=seg_params['submask_dilation'],  # a parameter that should be explored
        searcher=seg_params['searcher'],  # double this for SR
        radder=seg_params['radder'],  # double this for SR
        reg_iterations=seg_params['reg_iterations'], # fast test
        syn_sampling=seg_params['syn_sampling'],
        syn_metric=seg_params['syn_metric'],
        max_lab_plus_one=seg_params['max_lab_plus_one'],
        output_prefix=output_filename,
        verbose=seg_params['verbose'],
    )
probs = locseg['ljlf']['ljlf']['probabilityimages']
probability_labels = locseg['ljlf']['ljlf']['segmentation_numbers']
# find proper labels
whichprob75 = probability_labels.index(wlab[0])
whichprob76 = probability_labels.index(wlab[1])
probseg = ants.threshold_image(
  ants.resample_image_to_target(probs[whichprob75], srseg['super_resolution'] ) +
  ants.resample_image_to_target(probs[whichprob76], srseg['super_resolution'] ),
  0.3, 1.0 )
ants.image_write( probseg,  output_filename_sr_seg )
get_label_geo(
        probseg,
        srseg['super_resolution'],
        config.process_name,
        config.input_value,
        resolution="SR",
)
plot_output(
    srseg['super_resolution'],
    "outputs/basalforebrain-SR_ortho_plot.png",
    probseg,
)
handle_outputs(
    config.input_value,
    config.output_bucket,
    config.output_prefix,
    config.process_name,
    dev=True,
)
