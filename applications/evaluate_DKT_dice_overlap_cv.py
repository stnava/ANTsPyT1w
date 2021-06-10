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
from superiq import list_to_string


# get data from here https://ndownloader.figshare.com/files/26224727
tdir = "/Users/stnava/data/superiq_data_resources/"
if ( not path. exists( tdir ) ):
    raise RuntimeError('Failed to find the data directory')

brains = glob.glob(tdir+"segmentation_libraries/OASIS30/Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"segmentation_libraries/OASIS30/SegmentationsJLFOR/*")
brainsSeg.sort()
templatefilename = tdir + "template/adni_template.nii.gz"
templatesegfilename = tdir + "template/adni_template_dkt_labels.nii.gz"

seg_params={
    'submask_dilation': 8,
    'reg_iterations': [100, 100, 20],
    'searcher': 1,
    'radder': 2,
    'syn_sampling': 32,
    'syn_metric': 'mattes',
    'max_lab_plus_one': True, 'verbose': False}

seg_params_sr={
    'submask_dilation': seg_params['submask_dilation']*1,
    'reg_iterations': seg_params['reg_iterations'],
    'searcher': seg_params['searcher'],
    'radder': seg_params['radder'],
    'syn_sampling': seg_params['syn_sampling'],
    'syn_metric': seg_params['syn_metric'],
    'max_lab_plus_one': True, 'verbose': False}

sr_params={"upFactor": [2,2,2], "dilation_amount": seg_params["submask_dilation"], "verbose":True}
mdl = tf.keras.models.load_model("models/SEGSR_32_ANINN222_3.h5")

# store output data
brainName = []
# the three types of output which we will compute in series
dicevalNativeSeg = []
dicevalSRNativeSeg = []
dicevalSRSeg = []
dicevalSRSeg2 = []
wlab = [36,55,57] # for PPMI
wlab = [47,116,122,154,170] # eisai cortex
wlab = [75,76] # basal forebrain
evalfn='./dkt_eval' + list_to_string( wlab ) + '.csv'
print( "Labels:" + list_to_string( wlab ) + " " + evalfn, " : n : ", len( brains ) )
# full eval  here
for k in range( len(brainName), len( brains ) ):
    localid=os.path.splitext( os.path.splitext( os.path.basename( brains[k]) )[0])[0]
    print( str(k) + " " + localid)
    brainsLocal=brains.copy()
    brainsSegLocal=brainsSeg.copy()
    del brainsLocal[k:(k+1)]
    del brainsSegLocal[k:(k+1)]
    original_image = ants.image_read(brains[k])
    sloop = native_to_superres_ljlf_segmentation(
        target_image = original_image,
        segmentation_numbers = wlab,
        template = ants.image_read(templatefilename),
        template_segmentation = ants.image_read(templatesegfilename),
        library_intensity=images_to_list(brainsLocal),
        library_segmentation=images_to_list(brainsSegLocal),
        seg_params = seg_params,
        seg_params_sr = seg_params_sr,
        sr_params = sr_params,
        sr_model = mdl )

    # first - create a SR version of the image and the ground truth
    # NOTE: we binarize the labels
    # NOTE: the below call would only be used for evaluation ie when we have GT
    nativeGroundTruth = ants.image_read(brainsSeg[k])
    nativeGroundTruth = ants.mask_image( nativeGroundTruth, nativeGroundTruth, level = wlab, binarize=False )
    gtSR = super_resolution_segmentation_per_label(
            imgIn = ants.iMath( original_image, "Normalize"),
            segmentation = nativeGroundTruth, # usually, an estimate from a template, not GT
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose']
            )
    nativeGroundTruthProbSR = gtSR['probability_images'][0]
    nativeGroundTruthSR = gtSR['super_resolution_segmentation']
    nativeGroundTruthBinSR = ants.mask_image( nativeGroundTruthSR, nativeGroundTruthSR, wlab, binarize=True)

    # The full method involves:  (GT denotes ground truth)
    # [0.0] use template-based mapping to estimate initial labels
    # [1.0] run LJLF at native resolution (evaluate this wrt native res GT)
    #   [1.1] evaluate [1.0] wrt NN-Up-GT
    # [2.0] perform local simultaneous SR-Image and SR-Seg based on output of [1.0] (evaluate this wrt SR-GT)
    #   [2.1] evaluate [2.0] wrt NN-Up-GT
    # [3.0] run LJLF at SR based on [2.0] (evaluate this at SR wrt SR-GT)
    #   [3.1] evaluate [3.0] this wrt NN-Up-GT
    mypt = 0.5
    srsegLJLF = ants.threshold_image( sloop['srSeg']['probsum'], mypt, math.inf )
    nativeOverlapSloop = ants.label_overlap_measures( nativeGroundTruth, sloop['nativeSeg']['segmentation'] )
    srOnNativeOverlapSloop = ants.label_overlap_measures( nativeGroundTruthSR, sloop['srOnNativeSeg']['super_resolution_segmentation'] )
    srOverlapSloop = ants.label_overlap_measures( nativeGroundTruthSR, sloop['srSeg']['segmentation'] )
    srOverlap2 = ants.label_overlap_measures( nativeGroundTruthBinSR, srsegLJLF )
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
    df.to_csv( evalfn )
    ################################################################################
    # these are the outputs you would write out, along with label geometry for each segmentation
    ants.image_write( sloop['srOnNativeSeg']['super_resolution'], '/tmp/tempI.nii.gz' )
    ants.image_write( nativeGroundTruthSR, '/tmp/tempGT.nii.gz' )
    ants.image_write( sloop['srSeg']['segmentation'], '/tmp/tempSRSeg.nii.gz' )
    ants.image_write( sloop['nativeSeg']['segmentation'], '/tmp/tempORSeg.nii.gz' )
