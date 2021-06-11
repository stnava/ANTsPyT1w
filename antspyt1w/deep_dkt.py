import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

def deep_brain_parcellation(
    target_image,
    template,
    verbose=False,
):
    """

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

    atr = antspynet.deep_atropos(
        rigi,
        do_preprocessing=False,
        use_spatial_priors=1,
    )

    if verbose:
        print("End Atropos tissue segmentation")


    myk='segmentation_image'
    atr[myk] = ants.apply_transforms(
            target_image,
            atr[myk],
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )
    myk='probability_images'
    for myp in range( len( atr[myk] ) ):
        atr[myk][myp] = ants.apply_transforms(
            target_image,
            atr[myk][myp],
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )

    print("FIXME tissue propagation")

    return {"tissue_segmentation":atr,  "dkt_parcellation":dkt}
