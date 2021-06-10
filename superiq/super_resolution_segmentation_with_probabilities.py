import ants
import sys
import antspynet
import numpy as np
import tensorflow as tf

def super_resolution_segmentation_with_probabilities(
    img,
    initial_probabilities,
    sr_model_path,
    dilation_amount = 10
):
    """
    Simultaneous super-resolution and probabilistic segmentation.

    this function will perform localized super-resolution analysis on the
    underlying image content to produce a SR version of both the local region
    intentsity and segmentation probability.  the number of probability images
    determines the number of outputs for the intensity and probabilitiy list
    that are returned.  NOTE: should consider inverse logit transform option.

    Arguments
    ---------
    img : input intensity image

    initial_probabilities: original resolution probability images

    sr_model_path : the super resolution model - should have 2 channel input

    dilation_amount : amount the thresholded mask will be dilated to decide
    the SR region of interest

    Returns
    -------
    list of ANTsImages

    """

    mdl = tf.keras.models.load_model(sr_model_path)

    # SR Part
    srimglist = []
    srproblist = []
    mypt = 1.0 / len(initial_probabilities)

    for k in range(len(initial_probabilities)):
        temp = ants.threshold_image( initial_probabilities[k], mypt, 2.0 )
        tempm = ants.iMath(temp,'MD',dilation_amount)
        imgc = ants.crop_image(img,tempm)
        imgch = ants.crop_image(initial_probabilities[k],tempm)
#        print( "k-pre" + str(k) )
#        print( imgc.min() )
#        print( imgc.max() )
        imgcrescale = ants.iMath( imgc, "Normalize" ) * 255 - 127.5 # for SR
        imgchrescale = imgch * 255.0 - 127.5
        myarr = np.stack( [ imgcrescale.numpy(), imgchrescale.numpy() ],axis=3 )
        newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
        myarr = myarr.reshape( newshape )
        pred = mdl.predict( myarr )
        imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
        imgsr = ants.copy_image_info( imgc, imgsr )
        newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
        ants.set_spacing( imgsr,  newspc )
        imgsr = antspynet.regression_match_image( imgsr, ants.resample_image_to_target(imgc,imgsr) )
        imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
#        print( "k-post" + str(k) )
#        print( imgsr.min() )
#        print( imgsr.max() )
        imgsrh = ants.copy_image_info( imgc, imgsrh )
        ants.set_spacing( imgsrh,  newspc )
        tempup = ants.resample_image_to_target( temp, imgsr )
        srimglist.append( imgsr )
        # NOTE: get rid of pixellated junk/artifacts - acts like a prior
        srproblist.append( imgsrh * tempup )

    labels = {
        'sr_intensities':srimglist,
        'sr_probabilities':srproblist,
    }
    return labels
