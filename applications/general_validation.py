import os
# set number of threads - this should be optimized for your compute instance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
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
import multiprocessing as mp
import matplotlib.pyplot as plt
from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import images_to_list
from superiq import check_for_labels_in_image
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation
from superiq import native_to_superres_ljlf_segmentation
from superiq import list_to_string
from superiq.pipeline_utils import *

def run_example():
    # Change this
    tdir = "/home/ec2-user/superiq/validation"
    if ( not path. exists( tdir ) ):
        raise RuntimeError('Failed to find the data directory')

    print("====> Getting remote data")

    template_bucket = "invicro-pipeline-inputs"
    template_key = "adni_templates/adni_template.nii.gz"
    template_label_key = "adni_templates/adni_template_dkt_labels.nii.gz"
    model_bucket = "invicro-pipeline-inputs"
    model_key = "models/SEGSR_32_ANINN222_3.h5"
    atlas_bucket = "invicro-pipeline-inputs"
    atlas_image_prefix = "OASIS30/Brains/"
    atlas_label_prefix = "OASIS30/SegmentationsJLFOR/"


    template = get_s3_object(template_bucket, template_key, tdir)
    #template = ants.image_read(template)
    templateL = get_s3_object(template_bucket, template_label_key, tdir)
    #templateL = ants.image_read(templateL)

    model_path = get_s3_object(model_bucket, model_key, tdir)
    #mdl = tf.keras.models.load_model(model_path)

    atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
    brains = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_image_keys]
    brains.sort()
    #brains = [ants.image_read(i) for i in brains]

    atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
    brainsSeg = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_label_keys]
    brainsSeg.sort()
    #brainsSeg = [ants.image_read(i) for i in brainsSeg]


    # Will vary by tool
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

    sr_params={
        "upFactor": [2,2,2],
        "dilation_amount": seg_params["submask_dilation"],
        "verbose":True
    }

    # Define labels
    #wlab = [36,55,57] # for PPMI
    #wlab = [47,116,122,154,170] # eisai cortex
    wlab = [75,76] # basal forebrain

    # An example parameters argument
    native_to_superres_ljlf_segmentation_params = {
        "target_image": "",
        "segmentation_numbers": wlab,
        "template": template,
        "template_segmentation": templateL,
        "library_intensity": "",
        "library_segmentation": "",
        "seg_params": seg_params,
        "seg_params_sr": seg_params_sr,
        "sr_params": sr_params,
        "sr_model": model_path,
        "forward_transforms": None,
    }

    leave_one_out_cross_validation(
        native_to_superres_ljlf_segmentation,
        native_to_superres_ljlf_segmentation_params,
        brains,
        brainsSeg,
    )


def make_validation_pools(
    evaluation_function,
    evaluation_parameters,
    atlas_images_path,
    atlas_labels_path,
):
    #brainsLocal=[ants.image_read(i) for i in atlas_images_path]
    brainsLocal=atlas_images_path
    #brainsSegLocal=[ants.image_read(i) for i in atlas_labels_path]
    brainsSegLocal=atlas_labels_path
    pools = []
    for k in range(len(brainsLocal)):
        # Get the name of the target 
        localid=os.path.splitext(os.path.splitext(os.path.basename(brainsLocal[k]))[0])[0]
        print("Pool: " + str(k) + " " + localid)
        #brainsLocal=[ants.image_read(i) for i in atlas_images_path]
        #brainsSegLocal=[ants.image_read(i) for i in atlas_labels_path]
        original_image = brainsLocal[k]
        nativeGroundTruth = brainsSegLocal[k]
        # Remove the target brain from the atlas set 
        loo_brainsLocal = brainsLocal.copy()
        loo_brainsSegLocal = brainsSegLocal.copy()
        del loo_brainsLocal[k]
        del loo_brainsSegLocal[k]
        evaluation_parameters['target_image'] = original_image
        #evaluation_parameters['target_image_name'] = atlas_images_path[k]
        evaluation_parameters['nativeGroundTruth'] = nativeGroundTruth
        evaluation_parameters['library_intensity'] = loo_brainsLocal
        evaluation_parameters['library_segmentation'] = loo_brainsSegLocal
        evaluation_parameters['evaluation_function'] = evaluation_function
        pools.append(evaluation_parameters)
    return pools


def cross_validation(
    evaluation_parameters,
):
    evaluation_function = evaluation_parameters['evaluation_function']
    evaluation_parameters.pop('evaluation_function')
    target_image_name = evaluation_parameters['target_image']
    print(f'Evaluating: {target_image_name}')
    localid=os.path.splitext( os.path.splitext( os.path.basename( target_image_name))[0])[0]

    # Reparse dict after pickling
    evaluation_parameters['target_image'] = ants.image_read(evaluation_parameters['target_image'])
    evaluation_parameters['template'] = ants.image_read(evaluation_parameters['template'])
    evaluation_parameters['template_segmentation'] = \
        ants.image_read(evaluation_parameters['template_segmentation'])
    evaluation_parameters['library_intensity'] = \
        [ants.image_read(i) for i in evaluation_parameters['library_intensity']]
    evaluation_parameters['library_segmentation'] = \
        [ants.image_read(i) for i in evaluation_parameters['library_segmentation']]
    evaluation_parameters['sr_model'] = \
        tf.keras.models.load_model(evaluation_parameters['sr_model'])

    nativeGroundTruth = ants.image_read(evaluation_parameters['nativeGroundTruth'])
    evaluation_parameters.pop('nativeGroundTruth')

    # Main evaluation step
    sloop = evaluation_function(**evaluation_parameters)

    wlab = evaluation_parameters['segmentation_numbers']
    nativeGroundTruth = ants.mask_image(
        nativeGroundTruth,
        nativeGroundTruth,
        level = wlab,
        binarize=False
    )
    sr_params = evaluation_parameters['sr_params']
    mdl = evaluation_parameters['sr_model']
    original_image = evaluation_parameters['target_image']
    gtSR = super_resolution_segmentation_per_label(
            imgIn = ants.iMath( original_image, "Normalize"),
            segmentation = nativeGroundTruth, # usually, an estimate from a template, not GT
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = evaluation_parameters['segmentation_numbers'],
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose']
            )
    nativeGroundTruthProbSR = gtSR['probability_images'][0]
    nativeGroundTruthSR = gtSR['super_resolution_segmentation']
    nativeGroundTruthBinSR = ants.mask_image(
        nativeGroundTruthSR,
        nativeGroundTruthSR,
        wlab,
        binarize=True
    )

    mypt = 0.5
    srsegLJLF = ants.threshold_image( sloop['srSeg']['probsum'], mypt, math.inf )
    nativeOverlapSloop = ants.label_overlap_measures(
        nativeGroundTruth,
        sloop['nativeSeg']['segmentation']
    )
    srOnNativeOverlapSloop = ants.label_overlap_measures(
        nativeGroundTruthSR,
        sloop['srOnNativeSeg']['super_resolution_segmentation']
    )
    srOverlapSloop = ants.label_overlap_measures(
        nativeGroundTruthSR,
        sloop['srSeg']['segmentation']
    )
    srOverlap2 = ants.label_overlap_measures( nativeGroundTruthBinSR, srsegLJLF )

    brainName = []
    dicevalNativeSeg = []
    dicevalSRNativeSeg = []
    dicevalSRSeg = []
    dicevalSRSeg2 = []

    brainName.append( localid )
    dicevalNativeSeg.append(nativeOverlapSloop["MeanOverlap"][0])
    dicevalSRNativeSeg.append( srOnNativeOverlapSloop["MeanOverlap"][0])
    dicevalSRSeg.append( srOverlapSloop["MeanOverlap"][0])
    dicevalSRSeg2.append( srOverlap2["MeanOverlap"][0])
    record = {
        'name': brainName,
        'diceNativeSeg': dicevalNativeSeg,
        'diceSRNativeSeg': dicevalSRNativeSeg,
        'diceSRSeg': dicevalSRSeg }
    return record

def leave_one_out_cross_validation(
    evaluation_function,
    evaluation_parameters,
    atlas_images_path,
    atlas_labels_path,
    output_path = None,
    multiproc_pools = 1,
):
    print("====> Making processing pools")
    pools = make_validation_pools(
        evaluation_function,
        evaluation_parameters,
        atlas_images_path,
        atlas_labels_path,
    )
    print("====> Running cross validation")
    with mp.Pool(multiproc_pools) as p:
        results = p.map(cross_validation, pools)

    print("====> Combining results")
    records = [pd.DataFrame(i) for i in results]
    validation = pd.concat(records)
    wlab = evaluation_parameters['segmentation_numbers']
    if output_path is None:
        evalfn = 'dkt_eval' + list_to_string( wlab ) + '.csv'
    else:
        evalfn= output_path + '/dkt_eval' + list_to_string( wlab ) + '.csv'
    validation.to_csv(evalfn, index=False)


if __name__ == "__main__":
    run_example()

# these are the outputs you would write out, along with label geometry for each segmentation
#ants.image_write( sloop['srOnNativeSeg']['super_resolution'], '/tmp/tempI.nii.gz' )
#ants.image_write( nativeGroundTruthSR, '/tmp/tempGT.nii.gz' )
#ants.image_write( sloop['srSeg']['segmentation'], '/tmp/tempSRSeg.nii.gz' )
#ants.image_write( sloop['nativeSeg']['segmentation'], '/tmp/tempORSeg.nii.gz' )
