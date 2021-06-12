import os.path
from os import path

import ants
import antspynet
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import math

def dap( x ):
    bbt = ants.image_read( antspynet.get_antsxnet_data( "croppedMni152" ) )
    bbt = antspynet.brain_extraction( bbt, "t1" ) * bbt
    qaff=ants.registration( bbt, x, "AffineFast", aff_metric='GC', random_seed=1 )
    dapper = antspynet.deep_atropos( qaff['warpedmovout'], do_preprocessing=False )
    dappertox = ants.apply_transforms(
      x,
      dapper['segmentation_image'],
      qaff['fwdtransforms'],
      interpolator='genericLabel',
      whichtoinvert=[True]
    )
    return(  dappertox )

# this function looks like it's for BF but it can be used for any local label pair
def localsyn(img, template, hemiS, templateHemi, whichHemi, padder, iterations, output_prefix ):
    ihemi=img*ants.threshold_image( hemiS, whichHemi, whichHemi )
    themi=template*ants.threshold_image( templateHemi, whichHemi, whichHemi )
    hemicropmask = ants.threshold_image( templateHemi *
        ants.threshold_image( themi, 1e-3, math.inf),
        whichHemi, whichHemi ).iMath("MD",padder)
    tcrop = ants.crop_image( themi, hemicropmask  )
    syn = ants.registration( tcrop, ihemi, 'SyN', aff_metric='GC',
        syn_metric='CC', syn_sampling=2, reg_iterations=iterations,
        flow_sigma=3.0, total_sigma=0.50,
        verbose=False, outprefix = output_prefix, random_seed=1 )
    return syn


def hemi_reg(
    input_image,
    input_image_tissue_segmentation,
    input_image_hemisphere_segmentation,
    input_template,
    input_template_hemisphere_labels,
    output_prefix,
    padding=10,
    is_test=False ):
    """
    hemisphere focused registration that will produce jacobians and figures
    support data inspection

    input_image: input image

    input_image_tissue_segmentation: segmentation produced in ANTs style ie with
    labels defined by atropos brain segmentation (1 to 6)

    input_image_hemisphere_segmentation: left (1) and right (2) hemisphere
    segmentation

    input_template: template to which we register; prefer a population-specific
    relatively high-resolution template instead of MNI or biobank.

    input_template_hemisphere_labels: a segmentation image of template hemispheres
    with left labeled 1 and right labeled 2

    output_prefix: a path and prefix for registration related outputs

    padding: number of voxels to pad images, needed for diffzero

    is_test: boolean. this function can be long running by default. this would
    help testing more quickly by running fewer iterations.

    """

    img = ants.rank_intensity( input_image )
    ionlycerebrum = ants.threshold_image( input_image_tissue_segmentation, 2, 4 )

    tdap = dap( input_template )
    tonlycerebrum = ants.threshold_image( tdap, 2, 4 )
    maskinds=[2,3,4,5]
    template = ants.rank_intensity( input_template )

    regsegits=[200,200,20]

    # upsample the template if we are passing SR as input
    if min(ants.get_spacing(img)) < 0.8:
        regsegits=[200,200,200,20]
        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )

    if is_test:
        regsegits=[8,0,0]

    input_template_hemisphere_labels = ants.resample_image_to_target(
        input_template_hemisphere_labels,
        template,
        interp_type='genericLabel',
    )

    # now do a hemisphere focused registration
    synL = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=input_image_hemisphere_segmentation,
        templateHemi=input_template_hemisphere_labels,
        whichHemi=1,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "left_hemi_reg",
    )
    synR = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=input_image_hemisphere_segmentation,
        templateHemi=input_template_hemisphere_labels,
        whichHemi=2,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "right_hemi_reg",
    )

    ants.image_write(synL['warpedmovout'], output_prefix + "left_hemi_reg.nii.gz" )
    ants.image_write(synR['warpedmovout'], output_prefix + "right_hemi_reg.nii.gz" )

    fignameL = output_prefix + "left_hemi_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL)

    fignameR = output_prefix + "right_hemi_reg.png"
    ants.plot(synR['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameR)

    lhjac = ants.create_jacobian_determinant_image(
        synL['warpedmovout'],
        synL['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( lhjac, output_prefix+'left_hemi_jacobian.nii.gz' )

    rhjac = ants.create_jacobian_determinant_image(
        synR['warpedmovout'],
        synR['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( rhjac, output_prefix+'right_hemi_jacobian.nii.gz' )
    return {
        "synL":synL,
        "synR":synR,
        "lhjac":lhjac,
        "rhjac":rhjac
        }
