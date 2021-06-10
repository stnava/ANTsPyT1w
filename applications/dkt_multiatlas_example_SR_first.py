# this script assumes the image have been brain extracted
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import os.path
from os import path
import glob as glob
# set number of threads - this should be optimized per compute instance


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
from superiq import sort_library_by_similarity

# user definitions here
tdir = "/Users/stnava/code/super_resolution_pipelines/data/OASIS30/"
brains = glob.glob(tdir+"Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"Segmentations/*")
brainsSeg.sort()
templatefilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template.nii.gz"
templatesegfilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template_dkt_labels.nii.gz"

tdir = "/Users/stnava/Downloads/Hammers_mith-n30r95/"
brains = glob.glob(tdir+"*brain.nii.gz")
brains.sort()
brainsSeg = glob.glob(tdir+"*seg.nii.gz")
brainsSeg.sort()
tind = 28
templatefilename = brains[tind] # based on the return of sorting
templatesegfilename = brainsSeg[tind]

sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
output_filename = "outputs3/ADNI_Caudate"
wlab = [ 35, 39, 43 ] # caudate
# input data
imgIn = ants.image_read( infn )
template = ants.image_read( templatefilename )
templateL = ants.image_read( templatesegfilename )
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

havelabels = check_for_labels_in_image( wlab, templateL )
if not havelabels:
  raise Exception("Label missing from the template")

havelabels = check_for_labels_in_image( wlab, ants.image_read( brainsSeg[0] ) )
if not havelabels:
  raise Exception("Label missing from the library")

# expected output data
output_filename_or = output_filename + "_seg.nii.gz"
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"

if not 'reg' in locals():
    reg = ants.registration( imgIn, template, 'SyN', reg_iterations=(100,0,0), verbose=False )
    forward_transforms = reg['fwdtransforms']

initlab0 = ants.apply_transforms( imgIn, templateL, forward_transforms, interpolator="genericLabel" )
initlab0 = ants.mask_image( initlab0, initlab0, wlab )
initlab0b = ants.threshold_image( initlab0, 1, 1e9 ).morphology("dilate",3)
ants.plot_ortho( ants.crop_image( imgIn, initlab0b ), flat=True  )

if False:
    print("Sort start")
    mysim = sort_library_by_similarity( imgIn, initlab0, wlab, brains, brainsSeg )
    print("Sort finish")

doKM = False
if doKM:  # FIXME an alternative would be to pass in the extant segmentation
    maxk = 3
    kseg = ants.kmeans_segmentation( imgIn, k=maxk, mrf = 0)
    kseg = ants.threshold_image( kseg['segmentation'], 2, maxk )
    use_image = imgIn * kseg
else:
    use_image = ants.image_clone( imgIn )

doSR = True
if doSR:
    if not 'srseg' in locals():
        srseg = super_resolution_segmentation_per_label(
            imgIn = use_image,
            segmentation = initlab0,
            upFactor = [2,2,2],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = 12,
            verbose = True
        )
    initlab0 = ants.apply_transforms( srseg['super_resolution'], templateL,
        forward_transforms, interpolator="genericLabel" )
    ants.image_write( srseg['super_resolution'] , output_filename_sr )
    initlab0 = ants.mask_image( initlab0, initlab0, wlab )
    ants.image_write( initlab0 , output_filename_sr_seg_init )
    use_image = srseg['super_resolution']

locseg = ljlf_parcellation(
        use_image,
        segmentation_numbers=wlab,
        forward_transforms=forward_transforms,
        template=template,
        templateLabels=templateL,
        library_intensity = brains, # mysim['sorted_library_int'][0:12],
        library_segmentation =  brainsSeg, # mysim['sorted_library_seg'][0:12],
        submask_dilation=12,  # a parameter that should be explored
        searcher = 1,  # double this for SR
        radder   = 2,  # double this for SR
        reg_iterations=[50,50,10], # fast test
        syn_sampling = 32,
        syn_metric = 'mattes',
        max_lab_plus_one=True,
        verbose=True,
    )
################################################################
if doKM:
    locseg['segmentation'] = locseg['segmentation'] * ants.resample_image_to_target( kseg, locseg['segmentation'])

if doSR:
    ants.image_write( locseg['segmentation'], output_filename_sr_seg  ) #
else:
    ants.image_write( locseg['segmentation'], output_filename_or ) #

seggeo=ants.label_geometry_measures(locseg['segmentation'])
