import ants
import antspynet
import tensorflow as tf
import numpy as np
import os
import math

def t1_hypointensity( x, xWMProbability, template, templateWMPrior ):
    """
    generate input features supporting t1-based hypointensity algorithms

    input_image: input image; bias-corrected, brain-extracted and (potentially)
    registered to a template (uncertain if this is needed as yet)

    wmpriorIn: template-based tissue prior

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

    wmhunet.load_weights( antspyt1w.get_data("simwmhseg") )

    pp = wmhunet.predict( myfeatures )

    limg = ants.from_numpy( tf.squeeze( pp[0] ).numpy( ) )
    limg = ants.copy_image_info( templatesmall, limg )
    lesresam = ants.apply_transforms( x, limg, afftx, whichtoinvert=[False] )

    rnmdl = antspynet.create_resnet_model_3d( inshape,
      number_of_classification_labels = 1,
      layers = (1,2,3),
      residual_block_schedule = (3,4,6,3), squeezeAndExcite = True,
      lowestResolution = 32, cardinality = 1, mode = "regression" )
    rnmdl.load_weights( antspyt1w.get_data("simwmhdisc") )
    qq = rnmdl.predict( myfeatures )

    return {
        "wmh_probability_image":lesresam,
        "wmh_probability_of_existence":qq,
        "features":myfeatures }
