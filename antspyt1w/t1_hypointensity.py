import ants
import antspynet
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
from .get_data import get_data

def t1_hypointensity( x, xWMProbability, template, templateWMPrior, wmh_thresh=0.1 ):
    """
    provide measurements that may help decide if a given t1 image is likely
    to have hypointensity.

    input_image: input image; bias-corrected, brain-extracted and denoised

    wmpriorIn: template-based tissue prior

    wmh_thresh: float used to threshold WMH probability and produce summary data

    returns:
        - wmh_summary: summary data frame based on thresholding WMH probability at wmh_thresh
        - probability image denoting WMH probability; higher values indicate
          that WMH is more likely
        - an integral evidence that indicates the likelihood that the input
            image content supports the presence of white matter hypointensity.
            greater than zero is supportive of WMH.  the higher, the more so.
            less than zero is evidence against.

    """
    mybig = [88,128,128]
    templatesmall = ants.resample_image( template, mybig, use_voxels=True )
    qaff = ants.registration( x, templatesmall, 'SyN', aff_metric='GC', random_seed=1 )
    afftx = qaff['fwdtransforms'][1]
    templateWMPrior2x = ants.apply_transforms( x, templateWMPrior, qaff['fwdtransforms'] )
    realWM = ants.threshold_image( templateWMPrior2x , 0.9, math.inf )
    inimg = ants.rank_intensity( x )
    parcellateWMdnz = ants.kmeans_segmentation( inimg, 2, realWM, mrf=0.3 )['probabilityimages'][0]
    x2template = ants.apply_transforms( templatesmall, x, afftx, whichtoinvert=[True] )
    parcellateWMdnz2template = ants.apply_transforms( templatesmall, parcellateWMdnz, afftx, whichtoinvert=[True] )
    # features = rank+dnz-image, lprob, wprob, wprior at mybig resolution
    f1 = x2template.numpy()
    f2 = parcellateWMdnz2template.numpy()
    f3 = ants.apply_transforms( templatesmall, xWMProbability, afftx, whichtoinvert=[True] ).numpy()
    f4 = ants.apply_transforms( templatesmall, templateWMPrior, qaff['fwdtransforms'][0] ).numpy()
    myfeatures = np.stack( (f1,f2,f3,f4), axis=3 )
    newshape = np.concatenate( [ [1],np.asarray( myfeatures.shape )] )
    myfeatures = myfeatures.reshape( newshape )

    inshape = [None,None,None,4]
    wmhunet = antspynet.create_unet_model_3d( inshape,
        number_of_outputs = 1,
        number_of_layers = 4,
        mode = 'sigmoid' )

    wmhunet.load_weights( get_data("simwmhseg", target_extension='.h5') )

    pp = wmhunet.predict( myfeatures )

    limg = ants.from_numpy( tf.squeeze( pp[0] ).numpy( ) )
    limg = ants.copy_image_info( templatesmall, limg )
    lesresam = ants.apply_transforms( x, limg, afftx, whichtoinvert=[False] )

    rnmdl = antspynet.create_resnet_model_3d( inshape,
      number_of_classification_labels = 1,
      layers = (1,2,3),
      residual_block_schedule = (3,4,6,3), squeeze_and_excite = True,
      lowest_resolution = 32, cardinality = 1, mode = "regression" )
    rnmdl.load_weights( get_data("simwmdisc", target_extension='.h5' ) )
    qq = rnmdl.predict( myfeatures )

    lesresamb = ants.threshold_image( lesresam, wmh_thresh, 1.0 )
    lgo=ants.label_geometry_measures( lesresamb, lesresam )
    wmhsummary = pd.read_csv( get_data("wmh_evidence", target_extension='.csv' ) )
    wmhsummary.at[0,'Value']=lgo.at[0,'VolumeInMillimeters']
    wmhsummary.at[1,'Value']=lgo.at[0,'IntegratedIntensity']
    wmhsummary.at[2,'Value']=float(qq)

    return {
        "wmh_summary":wmhsummary,
        "wmh_probability_image":lesresam,
        "wmh_evidence_of_existence":float(qq),
        "wmh_max_prob":lesresam.max(),
        "features":myfeatures }
