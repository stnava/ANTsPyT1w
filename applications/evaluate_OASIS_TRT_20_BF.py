import os
# set number of threads - this should be optimized for your compute instance
mynt="16"
os.environ["TF_NUM_INTEROP_THREADS"] = mynt
os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt

import os.path
from os import path
import glob as glob

import math
import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import images_to_list
from superiq import check_for_labels_in_image
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation
from superiq import native_to_superres_ljlf_segmentation

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# get data from here https://ndownloader.figshare.com/files/26224727
tdir = "/Users/stnava/data/superiq_data_resources/"
if ( not path. exists( tdir ) ):
	raise RuntimeError('Failed to find the data directory')

brains = glob.glob(tdir+"segmentation_libraries/OASIS30/Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"segmentation_libraries/OASIS30/Segmentations/*")
brainsSeg.sort()
templatefilename = tdir + "template/adni_template.nii.gz"
templatesegfilename = tdir + "template/adni_template_dkt_labels.nii.gz"

if not 'low_volume' in locals():
    low_volume=False;

nhigh=20
if low_volume: # these subjects have very low volume BF relative to others
    brains=brains[nhigh:len(brains)]
    brainsSeg=brainsSeg[nhigh:len(brainsSeg)]
else:
    brains=brains[0:nhigh]
    brainsSeg=brainsSeg[0:nhigh]


seg_params={
    'submask_dilation': 8,
    'reg_iterations': [100, 100, 20],
    'searcher': 1,
    'radder': 2,
    'syn_sampling': 32,
    'syn_metric': 'mattes',
    'max_lab_plus_one': True, 'verbose': True}

seg_params_sr={
    'submask_dilation': seg_params['submask_dilation']*1,
    'reg_iterations': seg_params['reg_iterations'],
    'searcher': seg_params['searcher'],
    'radder': seg_params['radder'],
    'syn_sampling': seg_params['syn_sampling'],
    'syn_metric': seg_params['syn_metric'],
    'max_lab_plus_one': True, 'verbose': True}

sr_params={"upFactor": [2,2,2], "dilation_amount": seg_params["submask_dilation"], "verbose":True}
mdl = tf.keras.models.load_model("models/SEGSR_32_ANINN222_3.h5")

# store output data
brainName = []
# the three types of output which we will compute in series
dicevalNativeSeg = []
dicevalSRNativeSeg = []
dicevalSRSeg = []
dicevalSRSeg2 = []

brainsTest = glob.glob(tdir+"segmentation_libraries/OASISTRT20/Brains/*")
brainsTestSeg = glob.glob(tdir+"segmentation_libraries/OASISTRT20/Segmentations/*")
brainsTest.sort()
brainsTestSeg.sort()
for k in range( len(brainName), len( brainsTest ) ):
    localid=os.path.splitext( os.path.splitext( os.path.basename( brainsTest[k]) )[0])[0]
    original_image = ants.image_read(brainsTest[k])
    testseg = ants.image_read(brainsTestSeg[k])
    # map to basal forebrain
    testseg = ants.mask_image( testseg, testseg, [91,92], binarize=True )
    print( str(k) + " " + localid)
    # first - create a SR version of the image and the ground truth

    # now segment it with the library
    wlab = [75,76]
    sloop = native_to_superres_ljlf_segmentation(
        target_image = original_image,
        segmentation_numbers = wlab,
        template = ants.image_read(templatefilename),
        template_segmentation = ants.image_read(templatesegfilename),
        library_intensity=images_to_list(brains),
        library_segmentation=images_to_list(brainsSeg),
        seg_params = seg_params,
        sr_params = sr_params,
        sr_model = mdl )

    mypt = 0.5
    srGroundTruthNN = ants.resample_image_to_target( testseg, sloop['srOnNativeSeg']['super_resolution'] , interp_type='nearestNeighbor' )
    srsegLJLF = ants.threshold_image( sloop['srSeg']['probsum'], mypt, math.inf )
    nativejlf = ants.mask_image( sloop['nativeSeg']['segmentation'], sloop['nativeSeg']['segmentation'], wlab, binarize=True)
    nativeOverlapSloop = ants.label_overlap_measures( testseg, nativejlf )
    nativejlfsr = ants.mask_image( sloop['srOnNativeSeg']['super_resolution_segmentation'], sloop['srOnNativeSeg']['super_resolution_segmentation'], wlab, binarize=True)
    srOnNativeOverlapSloop = ants.label_overlap_measures( srGroundTruthNN, nativejlfsr )
    srjlf = ants.mask_image( sloop['srSeg']['segmentation'], sloop['srSeg']['segmentation'], wlab, binarize=True )
    srOverlapSloop = ants.label_overlap_measures( srGroundTruthNN, srjlf )
    srOverlap2 = ants.label_overlap_measures( srGroundTruthNN, srsegLJLF )
    # collect the 3 evaluation results - ready for data frame
    brainName.append( localid )
    dicevalNativeSeg.append(nativeOverlapSloop["MeanOverlap"][0])
    dicevalSRNativeSeg.append( srOnNativeOverlapSloop["MeanOverlap"][0])
    dicevalSRSeg.append( srOverlapSloop["MeanOverlap"][0])
    dicevalSRSeg2.append( srOverlap2["MeanOverlap"][0])
    print( brainName[k] + ": N: " + str(dicevalNativeSeg[k]) + " SRN: " +  str(dicevalSRNativeSeg[k])+ " SRN: " +  str(dicevalSRSeg[k]) )
    ################################################################################
    dict = {
        'name': brainName,
        'diceNativeSeg': dicevalNativeSeg,
        'diceSRNativeSeg': dicevalSRNativeSeg,
        'diceSRSeg': dicevalSRSeg }
    df = pd.DataFrame(dict)
    df.to_csv('./bf_sr_eval_TRT20_via_OASIS30.csv' )
    ################################################################################

# these are the outputs you would write out, along with label geometry for each segmentation
ants.image_write( sloop['srOnNativeSeg']['super_resolution'], '/tmp/tempI.nii.gz' )
ants.image_write( srGroundTruthNN, '/tmp/tempGT.nii.gz' )
ants.image_write( sloop['srSeg']['segmentation'], '/tmp/tempSRSeg.nii.gz' )
ants.image_write( sloop['nativeSeg']['segmentation'], '/tmp/tempORSeg.nii.gz' )
