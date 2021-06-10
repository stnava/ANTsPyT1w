import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

def deep_dkt(
    target_image,
    template,
):
    """

    """
    img = target_image

    rig = ants.registration( template, img, "AffineFast" )
    rigi = rig['warpedmovout']

    dkt = antspynet.desikan_killiany_tourville_labeling(
        rigi,
        do_preprocessing=False,
        return_probability_images=False,
        do_lobar_parcellation = True
    )

    for myk in dkt.keys():
        dkt[myk] = ants.apply_transforms(
            img,
            dkt[myk],
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )

    return dkt
