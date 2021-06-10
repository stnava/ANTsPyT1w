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
import glob as glob
import pandas as pd
import numpy as np
from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string


bbt = ants.image_read( antspynet.get_antsxnet_data( "biobank" ) )
bbt = antspynet.brain_extraction( bbt, "t1v0" ) * bbt

def dap( x ):
  qaff=ants.registration( bbt, x, "AffineFast" )
  dapper = antspynet.deep_atropos( qaff['warpedmovout'], do_preprocessing=False )
  dappertox = ants.apply_transforms( x, dapper['segmentation_image'], qaff['fwdtransforms'], interpolator='genericLabel', whichtoinvert=[True] )
  return(  dappertox )


prefix = '/tmp/temp_'
# FIXME: NOTE: this should work for either case SR or OR
img=ants.image_read( prefix+'SR.nii.gz' )
# img=ants.image_read( prefix+'cerebrum.nii.gz' ).resample_image( (128,128,128), use_voxels=True)
idap=ants.image_read( prefix+'tissueSegmentation.nii.gz' ).resample_image_to_target( img, interp_type='genericLabel')
hemiS=ants.image_read( prefix+'hemisphere.nii.gz' ).resample_image_to_target( img, interp_type='genericLabel')
citS=ants.image_read( prefix+'CIT168Labels.nii.gz' ).resample_image_to_target( img, interp_type='genericLabel')
bfprob1L=ants.image_read( prefix+'bfprob1left.nii.gz' ).resample_image_to_target( img, interp_type='linear')
bfprob1R=ants.image_read( prefix+'bfprob1right.nii.gz' ).resample_image_to_target( img, interp_type='linear')
bfprob2L=ants.image_read( prefix+'bfprob2left.nii.gz' ).resample_image_to_target( img, interp_type='linear')
bfprob2R=ants.image_read( prefix+'bfprob2right.nii.gz' ).resample_image_to_target( img, interp_type='linear')

tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
templatefn = tdir + "CIT168_T1w_700um_pad_adni0.nii.gz"
templateBF = glob.glob(tdir+"CIT168_basal_forebrain_adni_prob*gz")
templateBF.sort()
template = ants.image_read( templatefn )
# upsample the template if we are passing SR as input
if min(ants.get_spacing(img)) < 0.8:
    template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )
templateCIT = ants.image_read( tdir + "det_atlas_25_pad_LR_adni.nii.gz" ).resample_image_to_target( template, interp_type='genericLabel')
templateHemi = ants.image_read( tdir + "CIT168_T1w_700um_pad_HemisphereLabel_adni.nii.gz" ).resample_image_to_target( template, interp_type='genericLabel')

tdap = dap( template )
maskinds=[2,3,4,5]
temcerebrum = ants.mask_image(tdap,tdap,maskinds,binarize=True).iMath("GetLargestComponent")

# FIXME - this should be a "good" registration like we use in direct reg seg
# ideally, we would compute this separately - but also note that
regsegits=[200,200,200,50]


# now do a BF focused registration

# this function looks like it's for BF but it can be used for any local label pair
def localsyn( whichHemi, tbftotLoc, ibftotLoc, padder = 6 ):
    ihemi=img*ants.threshold_image( hemiS, whichHemi, whichHemi )
    themi=template*ants.threshold_image( templateHemi, whichHemi, whichHemi )
    rig = ants.registration( tbftotLoc, ibftotLoc, 'Rigid' )
    tbftotLoct = ants.threshold_image( tbftotLoc, 0.25, 2.0 ).iMath("MD", padder )
    tcrop = ants.crop_image( themi, tbftotLoct )
    syn = ants.registration( tcrop, img, 'SyNOnly',
        syn_metric='CC', syn_sampling=2, reg_iterations=(200,200,200),
        initial_transform=rig['fwdtransforms'][0], verbose=False)
    return syn


ibftotL = bfprob1L + bfprob2L
tbftotL = ( ants.image_read( templateBF[0] ) + ants.image_read( templateBF[2] ) ).resample_image_to_target( template, interp_type='linear')
ibftotR = bfprob1R + bfprob2R
tbftotR = ( ants.image_read( templateBF[1] ) + ants.image_read( templateBF[3] ) ).resample_image_to_target( template, interp_type='linear')
synL = localsyn( 1, tbftotL, ibftotL )
synR = localsyn( 2, tbftotR, ibftotR )
bftoiL1 = ants.apply_transforms( img, ants.image_read( templateBF[0] ).resample_image_to_target( template, interp_type='linear'), synL['invtransforms'] )
bftoiL2 = ants.apply_transforms( img, ants.image_read( templateBF[2] ).resample_image_to_target( template, interp_type='linear'), synL['invtransforms'] )
bftoiR1 = ants.apply_transforms( img, ants.image_read( templateBF[1] ).resample_image_to_target( template, interp_type='linear'), synR['invtransforms'] )
bftoiR2 = ants.apply_transforms( img, ants.image_read( templateBF[3] ).resample_image_to_target( template, interp_type='linear'), synR['invtransforms'] )

# FIXME - write out the above values to appropriately named files similar to templateBF names
# FIXME - get the volumes for each region (thresholded) and its sum
myspc = ants.get_spacing( img )
vbfL1 = np.asarray(myspc).prod() * bftoiL1.sum()
vbfL2 = np.asarray(myspc).prod() * bftoiL2.sum()
vbfR1 = np.asarray(myspc).prod() * bftoiR1.sum()
vbfR2 = np.asarray(myspc).prod() * bftoiR2.sum()

# same calculation but explicitly restricted to brain tissue
onlygm = ants.threshold_image( idap, 2, 4 )
vbfL1t = np.asarray(myspc).prod() * (bftoiL1*onlygm).sum()
vbfL2t = np.asarray(myspc).prod() * (bftoiL2*onlygm).sum()
vbfR1t = np.asarray(myspc).prod() * (bftoiR1*onlygm).sum()
vbfR2t = np.asarray(myspc).prod() * (bftoiR2*onlygm).sum()

ants.image_write( bftoiL1, prefix+'bfprob1leftSR.nii.gz' )
ants.image_write( bftoiR1, prefix+'bfprob1rightSR.nii.gz' )
ants.image_write( bftoiL2, prefix+'bfprob2leftSR.nii.gz' )
ants.image_write( bftoiR2, prefix+'bfprob2rightSR.nii.gz' )

