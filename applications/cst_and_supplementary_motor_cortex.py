# this script assumes the image have been brain extracted
import os.path
from os import path

# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import ants
import antspynet
import tensorflow as tf

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string

# input brain extracted target image
# inputs CIT168 and associated corticospinal tract labels
# output approximate CST and supplementary motor cortex
ifn = "/Users/stnava/data/data_old/PPMI/sr_examples/PPMI/3777/20160706/MRI_T1/I769284/direct_regseg/PPMI-3777-20160706-MRI_T1-I769284-direct_regseg-SR.nii.gz"
ifn = "/Users/stnava/data/data_old/PPMI/sr_examples/PPMI/18567/20150317/MRI_T1/I495172/direct_regseg/PPMI-18567-20150317-MRI_T1-I495172-direct_regseg-SR.nii.gz"
ifn = "/Users/stnava/Downloads/PPMI/sr_results_2021_03_29/regseg/PPMI/3190/20170504/MRI_T1/I901148/direct_regseg/PPMI-3190-20170504-MRI_T1-I901148-direct_regseg-SR.nii.gz"
tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
cstrfn = tdir + "CIT168_T1w_700um_pad_corticospinal_tract_right.nii.gz"
cstlfn = tdir + "CIT168_T1w_700um_pad_corticospinal_tract_left.nii.gz"
templatefn = tdir + "CIT168_T1w_700um_pad.nii.gz"
img = ants.image_read( ifn )
# img = ants.resample_image( img, (1,1,1) ) # FIXME - dont do this in real data
imgsmall = ants.resample_image( img, (128,128,128), use_voxels = True )
cstR = ants.image_read( cstrfn )
cstL = ants.image_read( cstlfn )
template = ants.image_read( templatefn )
# if a registration to CIT168 is already computed ( which it should be ) then
# just apply the resulting transformations otherwise compute a new reg
# existing registrations look like prefixWarp.nii.gz, prefixGenericAffine.mat
if not 'reg' in locals():
    reg = ants.registration( imgsmall, template, 'SyN' )

cstL2subject = ants.apply_transforms( img, cstL, reg['fwdtransforms'] )
cstR2subject = ants.apply_transforms( img, cstR, reg['fwdtransforms'] )


templateBB = antspynet.get_antsxnet_data( "biobank" )
templateBB = ants.image_read( templateBB )
templateBB = templateBB * antspynet.brain_extraction( templateBB )

if not 'rig' in locals():
    rig = ants.registration( templateBB, img, "Affine" )
    rigi = rig['warpedmovout']

if not 'dkt' in locals():
    dkt = antspynet.desikan_killiany_tourville_labeling( rigi,
      do_preprocessing=False,
      return_probability_images=False )

segorigspace = ants.apply_transforms( img, dkt, rig['fwdtransforms'],
    whichtoinvert=[True], interpolator='genericLabel')

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

if  False :
    bincst = ants.threshold_image( cstL2subject, 0.5, 1 )
    bincst = bincst + ants.threshold_image( cstR2subject, 0.5, 1 ) * ants.threshold_image( bincst, 0, 0 )
    mysma = mysma + ( bincst * ants.threshold_image( mysma, 0, 0 ) * 3. )
    ants.image_write( img, '/tmp/temp_i.nii.gz' )
    ants.image_write( mysma, '/tmp/temp_sma.nii.gz' )
    ants.image_write( segorigspace, '/tmp/temp_dkt.nii.gz' )
    ants.image_write( cstL2subject, '/tmp/temp_cstL.nii.gz' )
    ants.image_write( cstR2subject, '/tmp/temp_cstR.nii.gz' )
    derka

########################################
mysegnumbers = [ 1, 2 ] #
########################################
mdl = tf.keras.models.load_model( "models/SEGSR_32_ANINN222_bigTV3.h5" ) # FIXME - parameterize this

srseg = super_resolution_segmentation_per_label(
    imgIn = img,
    segmentation = mysma,
    upFactor = [2,2,2],
    sr_model = mdl,
    segmentation_numbers = mysegnumbers,
    dilation_amount = 6,
    verbose = True
)

ants.image_write( srseg['super_resolution'], '/tmp/temp_SRI.nii.gz' )
ants.image_write( srseg['super_resolution_segmentation'], '/tmp/temp_SRS.nii.gz' )

# FIXME - write out label geometry measures for:
# ants.threshold_image( CSTL, 0.5, 1 )
# ants.threshold_image( CSTR, 0.5, 1 ) and the SMA segmentation
# just do this at super-resolution
# also write out the full DKT image
