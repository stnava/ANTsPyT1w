import ants
import antspynet
import tensorflow as tf
import numpy as np
import os
import math

def brain_extraction( x ):
    """
    quick brain extraction for individual images

    x: input image

    """
    bxtmethod = 't1combined[5]' # better for individual subjects
    bxt = antspynet.brain_extraction( x, bxtmethod ).threshold_image(2,3).iMath("GetLargestComponent")
    return bxt


def label_hemispheres( x, template, templateLR ):
    """
    quick somewhat noisy registration solution to hemisphere labeling. typically
    we label left as 1 and right as 2.

    x: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    templateLR: a segmentation image of template hemispheres

    """
    reg = ants.registration( x, template, 'SyN', random_seed = 1 )
    return( ants.apply_transforms( x, templateLR, reg['fwdtransforms'],
        interpolator='genericLabel') )

def deep_tissue_segmentation( x, template=None, registration_map=None ):
    """
    modified slightly more efficient deep atropos that also handles the
    extra CSF issue.  returns segmentation and probability images. see
    the tissues csv available from get_data.

    x: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    registration_map: pre-existing output from ants.registration

    """
    if template is None:
        bbt = ants.image_read( antspynet.get_antsxnet_data( "biobank" ) )
        template = antspynet.brain_extraction( bbt, "t1" ) * bbt
        qaff=ants.registration( bbt, x, "AffineFast", aff_metric='GC', random_seed=1 )

    bbtr = ants.rank_intensity( template )
    if registration_map is None:
        registration_map = ants.registration( bbtr, x,
            "antsRegistrationSyNQuickRepro[a]",
            aff_iterations = (1500,500,0,0),
            random_seed=1 )

    mywarped = ants.apply_transforms( template, x,
        registration_map['fwdtransforms'] )

    dapper = antspynet.deep_atropos( mywarped,
        do_preprocessing=False, use_spatial_priors=1 )

    myk='segmentation_image'
    # the mysterious line below corrects for over-segmentation of CSF
    dapper[myk] = dapper[myk] * ants.threshold_image( mywarped, 1.0e-9, math.inf )
    dapper[myk] = ants.apply_transforms(
            x,
            dapper[myk],
            registration_map['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )

    myk='probability_images'
    myn = len( dapper[myk] )
    for myp in range( myn ):
        dapper[myk][myp] = ants.apply_transforms(
            x,
            dapper[myk][myp],
            registration_map['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='linear',
        )

    return dapper

def deep_brain_parcellation(
    target_image,
    template,
    do_cortical_propagation=False,
    verbose=False,
):
    """
    modified slightly more efficient deep dkt that also returns atropos output
    thus providing a complete hierarchical parcellation of t1w.  we run atropos
    here so we dont need to redo registration separately. see
    the lobes and dkt csv available from get_data.

    target_image: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    do_cortical_propagation: boolean, adds a bit extra time to propagate cortical
        labels explicitly into cortical segmentation

    verbose: boolean


    Returns
    -------
    a dictionary containing:

    - tissue_segmentation : 6 tissue segmentation
    - tissue_probabilities : probability images associated with above
    - dkt_parcellation : tissue agnostic DKT parcellation
    - dkt_lobes : major lobes of the brain
    - dkt_cortex: cortical tissue DKT parcellation (if requested)
    - hemisphere_labels: free to get hemisphere labels
    - wmSNR : white matter signal-to-noise ratio
    - wmcsfSNR : white matter to csf signal-to-noise ratio

    """
    if verbose:
        print("Begin registration")

    rig = ants.registration( template, target_image,
        "antsRegistrationSyNQuickRepro[a]",
        aff_iterations = (500,200,0,0),
        random_seed=1 )
    rigi = rig['warpedmovout']

    if verbose:
        print("Begin DKT")

    dkt = antspynet.desikan_killiany_tourville_labeling(
        rigi,
        do_preprocessing=False,
        return_probability_images=False,
        do_lobar_parcellation = True
    )

    for myk in dkt.keys():
        dkt[myk] = ants.apply_transforms(
            target_image,
            dkt[myk],
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )

    if verbose:
        print("Begin Atropos tissue segmentation")

    mydap = deep_tissue_segmentation(
        target_image,
        template,
        rig )

    if verbose:
        print("End Atropos tissue segmentation")

    myhemiL = ants.threshold_image( dkt['lobar_parcellation'], 1, 6 )
    myhemiR = ants.threshold_image( dkt['lobar_parcellation'], 7, 12 )
    myhemi = myhemiL + myhemiR * 2.0
    brainmask = ants.threshold_image( mydap['segmentation_image'], 1, 6 )
    myhemi = ants.iMath( brainmask, 'PropagateLabelsThroughMask', myhemi, 100, 0)

    cortprop = None
    if do_cortical_propagation:
        cortprop = ants.threshold_image( mydap['segmentation_image'], 2, 2 )
        cortlab = dkt['segmentation_image'] * ants.threshold_image( dkt['segmentation_image'], 1000, 5000  )
        cortprop = ants.iMath( cortprop, 'PropagateLabelsThroughMask',
            cortlab, 1, 0)

    wmseg = ants.threshold_image( mydap['segmentation_image'], 3, 3 )
    wmMean = target_image[ wmseg == 1 ].mean()
    wmStd = target_image[ wmseg == 1 ].std()
    csfseg = ants.threshold_image( mydap['segmentation_image'], 1, 1 )
    csfStd = target_image[ csfseg == 1 ].std()
    wmSNR = wmMean/wmStd
    wmcsfSNR = wmMean/csfStd

    return {
        "tissue_segmentation":mydap['segmentation_image'],
        "tissue_probabilities":mydap['probability_images'],
        "dkt_parcellation":dkt['segmentation_image'],
        "dkt_lobes":dkt['lobar_parcellation'],
        "dkt_cortex": cortprop,
        "hemisphere_labels": myhemi,
        "wmSNR": wmSNR,
        "wmcsfSNR": wmcsfSNR, }
