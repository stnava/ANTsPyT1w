# this script assumes the image have been brain extracted
import os.path
from os import path

threads = os.environ['cpu_threads']
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import tensorflow as tf
import sys

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string
from superiq.pipeline_utils import *

def main(input_config):
    c = LoadConfig(input_config)
    tdir = "data"
    ifn = get_pipeline_data(
        c.superres_suffix,
        c.input_value,
        c.pipeline_bucket,
        c.pipeline_prefix,
        tdir,
    )

    tfn = get_s3_object(c.template_bucket, c.template_key, tdir)
    tfnl = get_s3_object(c.template_bucket, c.template_label_key_left, tdir)
    tfnr = get_s3_object(c.template_bucket, c.template_label_key_right, tdir)
    cstrfn = tfnr
    cstlfn = tfnl
    templatefn = tfn
    img = ants.image_read( ifn )
    imgsmall = ants.resample_image( img, (128,128,128), use_voxels = True )
    cstR = ants.image_read( cstrfn )
    cstL = ants.image_read( cstlfn )
    template = ants.image_read( templatefn )
    reg = ants.registration( imgsmall, template, 'SyN' )

    cstL2subject = ants.apply_transforms( img, cstL, reg['fwdtransforms'] )
    cstR2subject = ants.apply_transforms( img, cstR, reg['fwdtransforms'] )

    templateBB = antspynet.get_antsxnet_data( "biobank" )
    templateBB = ants.image_read( templateBB )
    templateBB = templateBB * antspynet.brain_extraction( templateBB )

    rig = ants.registration( templateBB, img, "Affine" )
    rigi = rig['warpedmovout']

    dkt = antspynet.desikan_killiany_tourville_labeling(
        rigi,
        do_preprocessing=False,
        return_probability_images=False,
    )

    segorigspace = ants.apply_transforms(
        img,
        dkt,
        rig['fwdtransforms'],
        whichtoinvert=[True],
        interpolator='genericLabel',
    )

    # This is just an estimate - unclear what these parameters should be:
    #    both 2028 and 2017 or just 2028?  superior frontal + paracentral
    #    should i use a "low" threshold (0.05) in addition to dilation?
    dterm = 4
    mysmaL = ants.threshold_image( cstL2subject, 0.05, 2.0 ).iMath("MD",dterm) * (
        ants.threshold_image( segorigspace, 2028, 2028 ) +
        ants.threshold_image( segorigspace, 2017, 2017 ) )
    mysmaR = ants.threshold_image( cstR2subject, 0.05, 2.0 ).iMath("MD",dterm) * (
        ants.threshold_image( segorigspace, 1028, 1028 ) +
        ants.threshold_image( segorigspace, 1017, 1017 ) )
    mysma = mysmaL + mysmaR * 2.

    bincst = ants.threshold_image( cstL2subject, 0.5, 1 )
    bincst = bincst + \
        ants.threshold_image( cstR2subject, 0.5, 1 ) * ants.threshold_image( bincst, 0, 0 )
    mysma = mysma + ( bincst * ants.threshold_image( mysma, 0, 0 ) * 3. )

    mysegnumbers = c.wlab
    model = get_s3_object(c.model_bucket, c.model_key, tdir)
    mdl = tf.keras.models.load_model(model)
    srseg = super_resolution_segmentation_per_label(
        imgIn = img,
        segmentation = mysma,
        upFactor = [2,2,2],
        sr_model = mdl,
        segmentation_numbers = mysegnumbers,
        dilation_amount = 6,
        verbose = True
    )
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    ants.image_write( img, 'outputs/temp_i.nii.gz' )
    ants.image_write( mysma, 'outputs/temp_sma.nii.gz' )
    ants.image_write( segorigspace, 'outputs/temp_dkt.nii.gz' )
    ants.image_write( cstL2subject, 'outputs/temp_cstL.nii.gz' )
    ants.image_write( cstR2subject, 'outputs/temp_cstR.nii.gz' )
    ants.image_write( srseg['super_resolution'], 'outputs/temp_SRI.nii.gz' )
    ants.image_write( srseg['super_resolution_segmentation'], 'outputs/temp_SRS.nii.gz' )

    cstlT = ants.threshold_image( cstL2subject, 0.5, 1 )
    cstl_df = ants.label_geometry_measures(cstlT, cstlT)
    cstl_df.to_csv('outputs/cst_left_OR.csv', index=False)

    cstrT = ants.threshold_image( cstR2subject, 0.5, 1 )
    cstr_df = ants.label_geometry_measures(cstrT, cstrT)
    cstr_df.to_csv('outputs/cst_right_OR.csv', index=False)

    sr_segT = ants.threshold_image( srseg['super_resolution_segmentation'], 0.5, 1 )
    sr_seg_df = ants.label_geometry_measures(sr_segT, sr_segT)
    sr_seg_df.to_csv('outputs/seg_SR.csv', index=False)

    sma = ants.threshold_image( mysma, 0.5, 1 )
    sma_df = ants.label_geometry_measures(sma,sma)
    sma_df.to_csv('outputs/sma_labels.csv', index=False)

    handle_outputs(
        c.input_value,
        c.output_bucket,
        c.output_prefix,
        c.process_name,
    )

    # FIXME - write out label geometry measures for:
    # ants.threshold_image( CSTL, 0.5, 1 )
    # ants.threshold_image( CSTR, 0.5, 1 ) and the SMA segmentation
    # just do this at super-resolution
    # also write out the full DKT image

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
