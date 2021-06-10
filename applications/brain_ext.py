# this script assumes the image have been brain extracted
import os.path
from os import path

threads = "8"
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import tensorflow as tf
import os
import sys
import pandas as pd
import numpy as np

from superiq.pipeline_utils import *
# 3366/MPRAGE_T1_SAG/2011-08-17_11_16_19.0/S124461/PPMI_3366_MR_MPRAGE_T1_SAG__br_raw_20111005172139820_3_S124461_I259611.nii"
rdir="/Users/stnava/Downloads/PPMI-4/"
image_path=rdir+"60024/MPRAGE_GRAPPA/2014-02-25_09_58_51.0/S225648/PPMI_60024_MR_MPRAGE_GRAPPA__br_raw_20140723142403493_65_S225648_I436437.nii"
image_path=rdir+"3366/MPRAGE_T1_SAG/2011-08-17_11_16_19.0/S124461/PPMI_3366_MR_MPRAGE_T1_SAG__br_raw_20111005172139820_3_S124461_I259611.nii"
image_path=rdir+"3126/MPRAGE_GRAPPA/2013-09-18_10_59_00.0/S205747/PPMI_3126_MR_MPRAGE_GRAPPA__br_raw_20131108092457665_121_S205747_I397751.nii"
image_path=rdir+"3421/SAG_T1_3D_FSPGR/2011-06-15_08_19_11.0/S113573/PPMI_3421_MR_SAG_T1_3D_FSPGR__br_raw_20110705134018388_23_S113573_I243259.nii"
input_image = ants.image_read(image_path)
ukbb = antspynet.get_antsxnet_data("biobank")
template = ants.image_read(ukbb)
btem = antspynet.brain_extraction(template, 't1')
template = template * btem
run_extra=True

def reg_bxt( intemplate, inimg, inbxt, bxt_type, txtype, dilation=0 ):
    inbxtdil = ants.iMath( inbxt, "MD", dilation )
    img = ants.iMath( inimg * inbxt, "TruncateIntensity", 0.0001, 0.999)
    imgn4 = ants.n3_bias_field_correction(img, downsample_factor=4)
    rig = ants.registration(
            intemplate,
            imgn4,
            txtype,
            aff_iterations=(10000, 500, 0, 0),
        )
    if dilation > 0:
        rigi = ants.apply_transforms( intemplate, inimg * inbxtdil, rig['fwdtransforms'] )
    else:
        rigi = ants.apply_transforms( intemplate, inimg, rig['fwdtransforms'] )
    rigi = ants.iMath( rigi, "Normalize")
    rigi = ants.n3_bias_field_correction( rigi, downsample_factor=4 )
    bxt = antspynet.brain_extraction(rigi, bxt_type )
    if bxt_type == 't1combined':
        bxt = ants.threshold_image( bxt, 2, 3 )
    bxt = ants.apply_transforms(
            fixed=inimg,
            moving=bxt,
            transformlist=rig['invtransforms'],
            whichtoinvert=[True,],
        )
    return bxt

b0 = antspynet.brain_extraction(input_image, 't1')
rbxt1 = reg_bxt( template, input_image, b0, 't1', 'Rigid', dilation=0 )
rbxt2 = reg_bxt( template, input_image, rbxt1, 't1', 'Rigid', dilation=0  )
rbxt3 = reg_bxt( template, input_image, rbxt2, 't1', 'Rigid', dilation=0 )
rbxt3 = ants.threshold_image( rbxt3, 0.5, 2. ).iMath("GetLargestComponent")
rbxt4 = reg_bxt( template, input_image, rbxt3, 't1combined', 'Rigid', dilation=0 )
if run_extra:
    rbxt5 = reg_bxt( template, input_image, rbxt4, 't1combined', 'Rigid', dilation=25 )
    img = ants.iMath(input_image * rbxt5, "TruncateIntensity", 0.0001, 0.999)
    imgn4 = ants.n4_bias_field_correction(img, shrink_factor=4)
    syn=ants.registration(template, imgn4, "SyN" )
    bxt = ants.apply_transforms( imgn4, btem, syn['invtransforms'], interpolator='nearestNeighbor')
    bxt = bxt * rbxt5
else:
    bxt = rbxt4

img = ants.iMath(input_image * bxt, "TruncateIntensity", 0.0001, 0.999)
imgn4 = ants.n4_bias_field_correction(img, shrink_factor=4)

# write out bxt and n4 image
ants.plot( imgn4, axis=0)
