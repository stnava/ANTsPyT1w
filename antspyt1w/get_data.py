"""
Get local ANTsPyT1w data
"""

__all__ = ['get_data','map_segmentation_to_dataframe','hierarchical',
    'random_basis_projection', 'deep_dkt','deep_hippo','deep_tissue_segmentation',
    'deep_brain_parcellation', 'deep_mtl', 'label_hemispheres','brain_extraction',
    'hemi_reg', 'region_reg', 't1_hypointensity', 'zoom_syn',
    'map_intensity_to_dataframe', 'deep_nbm', 'map_cit168']

from pathlib import Path
import os
import pandas as pd
import math
import os.path
from os import path
import pickle
import sys
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds

import ants
import antspynet
import tensorflow as tf

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspyt1w/')

def get_data( name=None, force_download=False, version=30, target_extension='.csv' ):
    """
    Get ANTsPyT1w data filename

    The first time this is called, it will download data to ~/.antspyt1w.
    After, it will just read data from disk.  The ~/.antspyt1w may need to
    be periodically deleted in order to ensure data is current.

    Arguments
    ---------
    name : string
        name of data tag to retrieve
        Options:
            - 'all'
            - 'dkt'
            - 'hemisphere'
            - 'lobes'
            - 'tissues'
            - 'T_template0'
            - 'T_template0_LR'
            - 'T_template0_LobesBstem'
            - 'T_template0_WMP'
            - 'T_template0_Symmetric'
            - 'T_template0_SymmetricLR'
            - 'PPMI-3803-20120814-MRI_T1-I340756'
            - 'simwmseg'
            - 'simwmdisc'
            - 'wmh_evidence'
            - 'wm_major_tracts'

    force_download: boolean

    version: version of data to download (integer)

    Returns
    -------
    string
        filepath of selected data

    Example
    -------
    >>> import ants
    >>> ppmi = ants.get_ants_data('ppmi')
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def download_data( version ):
        url = "https://ndownloader.figshare.com/articles/14766102/versions/" + str(version)
        target_file_name = "14766102.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )
        os.remove( DATA_PATH + target_file_name )

    if force_download:
        download_data( version = version )


    files = []
    for fname in os.listdir(DATA_PATH):
        if ( fname.endswith(target_extension) ) :
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len( files ) == 0 :
        download_data( version = version )
        for fname in os.listdir(DATA_PATH):
            if ( fname.endswith(target_extension) ) :
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    if name == 'all':
        return files

    datapath = None

    for fname in os.listdir(DATA_PATH):
        mystem = (Path(fname).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        if ( name == mystem and fname.endswith(target_extension) ) :
            datapath = os.path.join(DATA_PATH, fname)

    if datapath is None:
        os.listdir(DATA_PATH)
    return datapath



def map_segmentation_to_dataframe( segmentation_type, segmentation_image ):
    """
    Match the segmentation to its appropriate data frame.  We do not check
    if the segmentation_type and segmentation_image match; this may be indicated
    by the number of missing values on output eg in column VolumeInMillimeters.

    Arguments
    ---------
    segmentation_type : string
        name of segmentation_type data frame to retrieve
        Options:
            - 'dkt'
            - 'lobes'
            - 'tissues'
            - 'hemisphere'
            - 'wm_major_tracts'

    segmentation_image : antsImage with same values (or mostly the same) as are
        expected by segmentation_type

    Returns
    -------
    dataframe

    """
    mydf_fn = get_data( segmentation_type )
    mydf = pd.read_csv( mydf_fn )
    mylgo = ants.label_geometry_measures( segmentation_image )
    return pd.merge( mydf, mylgo, how='left', on=["Label"] )

def map_intensity_to_dataframe( segmentation_type, intensity_image, segmentation_image ):
    """
    Match itensity values within segmentation labels to its appropriate data frame.

    Arguments
    ---------
    segmentation_type : string
        name of segmentation_type data frame to retrieve
        Options:
            - see get_data function or ~/.antspyt1w folder
            - e.g. lobes

    intensity_image : antsImage with intensity values to summarize

    segmentation_image : antsImage with same values (or mostly the same) as are
        expected by segmentation_type

    Returns
    -------
    dataframe

    """
    mydf_fn = get_data( segmentation_type )
    mydf = pd.read_csv( mydf_fn )
    mylgo = ants.label_stats( intensity_image, segmentation_image )
    mylgo = mylgo.rename(columns = {'LabelValue':'Label'})
    return pd.merge( mydf, mylgo, how='left', on=["Label"] )



def myproduct(lst):
    return( functools.reduce(mul, lst) )


def random_basis_projection( x, template, type_of_transform='Similarity',
    nBasis=10, random_state = 99 ):
    """
    Produce unbiased data descriptors for a given image which can be used
    to assist data inspection and ranking.  can be used with any image
    brain extracted or not, any modality etc.   but we assume we can
    meaningfully map to a template, at least with a low-dimensional
    transformation, e.g. Translation, Rigid, Similarity.

    Arguments
    ---------
    x : antsImage

    template : antsImage reference template

    type_of_transform: one of Translation, Rigid, Similarity, Affine

    nBasis : number of variables to derive

    random_state : seed

    Returns
    -------
    dataframe with projections

    """
    np.random.seed(int(random_state))
    nvox = template.shape
    X = np.random.rand( nBasis+1, myproduct( nvox ) )
    u, s, randbasis = svds(X, k=nBasis)
    if randbasis.shape[1] != myproduct(nvox):
        raise ValueError("columns in rand basis do not match the nvox product")

    randbasis = np.transpose( randbasis )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    norm = ants.rank_intensity(x)
    resamp = ants.registration( template, norm,
        type_of_transform=type_of_transform,
        aff_metric='GC', random_seed=1 )['warpedmovout']
    imat = ants.get_neighborhood_in_mask(resamp, resamp*0+1,[0,0,0], boundary_condition='mean' )
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    record = {}
    uproj_counter = 0
    for i in uproj[0]:
        uproj_counter += 1
        name = "RandBasisProj" + str(uproj_counter).zfill(2)
        record[name] = i
    uprojpos_counter = 0
    for i in uprojpos[0]:
        uprojpos_counter += 1
        name = "RandBasisProjPos" + str(uprojpos_counter).zfill(2)
        record[name] = i
    df = pd.DataFrame(record, index=[0])
    return df



def brain_extraction( x ):
    """
    quick brain extraction for individual images

    x: input image

    """
    bxtmethod = 't1combined[5]' # better for individual subjects
    bxt = antspynet.brain_extraction( x, bxtmethod ).threshold_image(2,3).iMath("GetLargestComponent")
    return bxt


def label_hemispheres( x, template, templateLR, reg_iterations=[200,50,2,0] ):
    """
    quick somewhat noisy registration solution to hemisphere labeling. typically
    we label left as 1 and right as 2.

    x: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    templateLR: a segmentation image of template hemispheres

    reg_iterations: reg_iterations for ants.registration

    """
    reg = ants.registration(
        ants.rank_intensity(x),
        ants.rank_intensity(template),
        'SyN',
        aff_metric='GC',
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=reg_iterations,
        random_seed = 1 )
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
        qaff=ants.registration( bbt, ants.rank_intensity(x), "AffineFast", aff_metric='GC', random_seed=1 )

    bbtr = ants.rank_intensity( template )
    if registration_map is None:
        registration_map = ants.registration(
            bbtr,
            ants.rank_intensity(x),
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

    rig = ants.registration( template, ants.rank_intensity(target_image),
        "antsRegistrationSyNQuickRepro[a]",
        aff_iterations = (500,200,0,0),
        random_seed=1 )
    rigi = ants.apply_transforms( template, target_image, rig['fwdtransforms'])

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


def deep_hippo(
    img,
    template,
    number_of_tries = 10,
):

    avgleft = img * 0
    avgright = img * 0
    for k in range(number_of_tries):
        rig = ants.registration( template, ants.rank_intensity(img),
            "antsRegistrationSyNQuickRepro[a]", random_seed=k )
        rigi = ants.apply_transforms( template, img, rig['fwdtransforms'] )
        hipp = antspynet.hippmapp3r_segmentation( rigi, do_preprocessing=False )
        hippr = ants.apply_transforms(
            img,
            hipp,
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )
        avgleft = avgleft + ants.threshold_image( hippr, 2, 2 ) / float(number_of_tries)
        avgright = avgright + ants.threshold_image( hippr, 1, 1 ) / float(number_of_tries)


    avgright = ants.iMath(avgright,"Normalize")  # output: probability image right
    avgleft = ants.iMath(avgleft,"Normalize")    # output: probability image left
    hippright_bin = ants.threshold_image( avgright, 0.5, 2.0 ).iMath("GetLargestComponent")
    hippleft_bin = ants.threshold_image( avgleft, 0.5, 2.0 ).iMath("GetLargestComponent")

    hippleftORlabels  = ants.label_geometry_measures(hippleft_bin, avgleft)
    hippleftORlabels['Description'] = 'left hippocampus'
    hipprightORlabels  = ants.label_geometry_measures(hippright_bin, avgright)
    hipprightORlabels['Description'] = 'right hippocampus'

    labels = {
        'HLProb':avgleft,
        'HLBin':hippleft_bin,
        'HLStats': hippleftORlabels,
        'HRProb':avgright,
        'HRBin':hippright_bin,
        'HRStats': hipprightORlabels,
    }
    return labels


def dap( x ):
    bbt = ants.image_read( antspynet.get_antsxnet_data( "croppedMni152" ) )
    bbt = antspynet.brain_extraction( bbt, "t1" ) * bbt
    qaff=ants.registration( bbt, ants.rank_intensity(x), "AffineFast", aff_metric='GC', random_seed=1 )
    qaff['warpedmovout'] = ants.apply_transforms( bbt, x, qaff['fwdtransforms'] )
    dapper = antspynet.deep_atropos( qaff['warpedmovout'], do_preprocessing=False )
    dappertox = ants.apply_transforms(
      x,
      dapper['segmentation_image'],
      qaff['fwdtransforms'],
      interpolator='genericLabel',
      whichtoinvert=[True]
    )
    return(  dappertox )

def deep_mtl(t1):

    """
    Hippocampal/Enthorhinal segmentation using "Deep Flash"

    Perform hippocampal/entorhinal segmentation in T1 images using
    labels from Mike Yassa's lab

    https://faculty.sites.uci.edu/myassa/

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA3
    Label 14:  right DG/CA3
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

    """

    verbose = False

    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    label_descriptions = ['background',
                          'left aLEC',
                          'right aLEC',
                          'left pMEC',
                          'right pMEC',
                          'left perirhinal',
                          'right perirhinal',
                          'left parahippocampal',
                          'right parahippocampal',
                          'left DG/CA3',
                          'right DG/CA3',
                          'left CA1',
                          'right CA1',
                          'left subiculum',
                          'right subiculum'
                          ]

    template = ants.image_read(antspynet.get_antsxnet_data("deepFlashTemplateT1SkullStripped"))
    registration = ants.registration(fixed=template, moving=t1,
        type_of_transform="antsRegistrationSyNQuickRepro[a]", verbose=verbose)
    template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                               invtransforms=registration['invtransforms'])
    t1_warped = registration['warpedmovout']

    df = antspynet.deep_flash(t1_warped, do_preprocessing=False, verbose=verbose)

    probability_images = list()
    for i in range(len(df['probability_images'])):
        probability_image = ants.apply_transforms(fixed=t1,
                                                  moving=df['probability_images'][i],
                                                  transformlist=template_transforms['invtransforms'],
                                                  whichtoinvert=[True],
                                                  interpolator="linear",
                                                  verbose=verbose)
        probability_images.append(probability_image)

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    mtl_description = pd.DataFrame(labels, columns=['Label'])
    mtl_description.insert(1, "Description", label_descriptions)

    mtl_description = map_segmentation_to_dataframe( 'mtl_description', relabeled_image )

    deep_mtl_dictionary = {
                          'mtl_description':mtl_description,
                          'mtl_segmentation':relabeled_image,
                          'mtl_probability_images':probability_images
                          }
    return(deep_mtl_dictionary)

# this function looks like it's for BF but it can be used for any local label pair
def localsyn(img, template, hemiS, templateHemi, whichHemi, padder, iterations,
    output_prefix, total_sigma=0.5 ):
    ihemi=img*ants.threshold_image( hemiS, whichHemi, whichHemi )
    themi=template*ants.threshold_image( templateHemi, whichHemi, whichHemi )
    loquant = np.quantile(themi.numpy(),0.01)+1e-6 # identify background value and add epsilon to it
    hemicropmask = ants.threshold_image( templateHemi *
        ants.threshold_image( themi, loquant, math.inf),
        whichHemi, whichHemi ).iMath("MD",padder)
    tcrop = ants.crop_image( themi, hemicropmask  )
    syn = ants.registration( tcrop, ihemi, 'SyN', aff_metric='GC',
        syn_metric='CC', syn_sampling=2, reg_iterations=iterations,
        flow_sigma=3.0, total_sigma=total_sigma,
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
    labels_to_register = [2,3,4,5],
    total_sigma=0.5,
    is_test=False ):
    """
    hemisphere focused registration that will produce jacobians and figures to
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

    labels_to_register: list of integer segmentation labels to use to define
    the tissue types / regions of the brain to register.

    total_sigma: scalar >= 0.0 ; higher means more constrained registration.

    is_test: boolean. this function can be long running by default. this would
    help testing more quickly by running fewer iterations.

    """

    img = ants.rank_intensity( input_image )
    ionlycerebrum = ants.mask_image( input_image_tissue_segmentation,
        input_image_tissue_segmentation, labels_to_register, 1 )

    tdap = dap( input_template )
    tonlycerebrum = ants.mask_image( tdap, tdap, labels_to_register, 1 )
    template = ants.rank_intensity( input_template )

    regsegits=[200,200,20]

#    # upsample the template if we are passing SR as input
#    if min(ants.get_spacing(img)) < 0.8 :
#        regsegits=[200,200,200,20]
#        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )
#        tonlycerebrum = ants.resample_image_to_target( tonlycerebrum,
#            template,
#            interp_type='genericLabel',
#        )

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
        total_sigma=total_sigma,
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
        total_sigma=total_sigma,
    )

    ants.image_write(synL['warpedmovout'], output_prefix + "left_hemi_reg.nii.gz" )
    ants.image_write(synR['warpedmovout'], output_prefix + "right_hemi_reg.nii.gz" )

    fignameL = output_prefix + "_left_hemi_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL, black_bg=False, crop=True )

    fignameR = output_prefix + "_right_hemi_reg.png"
    ants.plot(synR['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameR, black_bg=False, crop=True )

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
        "synLpng":fignameL,
        "synR":synR,
        "synRpng":fignameR,
        "lhjac":lhjac,
        "rhjac":rhjac
        }



def region_reg(
    input_image,
    input_image_tissue_segmentation,
    input_image_region_segmentation,
    input_template,
    input_template_region_segmentation,
    output_prefix,
    padding=10,
    labels_to_register = [2,3,4,5],
    total_sigma=0.5,
    is_test=False ):
    """
    region focused registration that will produce jacobians and figures to
    support data inspection.  region-defining images should be binary.

    input_image: input image

    input_image_tissue_segmentation: segmentation produced in ANTs style ie with
    labels defined by atropos brain segmentation (1 to 6)

    input_image_region_segmentation: a local region to register - binary.

    input_template: template to which we register; prefer a population-specific
    relatively high-resolution template instead of MNI or biobank.

    input_template_region_segmentation: a segmentation image of template regions - binary.

    output_prefix: a path and prefix for registration related outputs

    padding: number of voxels to pad images, needed for diffzero

    labels_to_register: list of integer segmentation labels to use to define
    the tissue types / regions of the brain to register.

    total_sigma: scalar >= 0.0 ; higher means more constrained registration.

    is_test: boolean. this function can be long running by default. this would
    help testing more quickly by running fewer iterations.

    """

    img = ants.rank_intensity( input_image )
    ionlycerebrum = ants.mask_image( input_image_tissue_segmentation,
        input_image_tissue_segmentation, labels_to_register, 1 )

    tdap = dap( input_template )
    tonlycerebrum = ants.mask_image( tdap, tdap, labels_to_register, 1 )
    template = ants.rank_intensity( input_template )

    regsegits=[200,200,20]

    # upsample the template if we are passing SR as input
    if min(ants.get_spacing(img)) < 0.8:
        regsegits=[200,200,200,20]
        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )
        tonlycerebrum = ants.resample_image_to_target( tonlycerebrum,
            template,
            interp_type='genericLabel',
        )

    if is_test:
        regsegits=[8,0,0]

    input_template_region_segmentation = ants.resample_image_to_target(
        input_template_region_segmentation,
        template,
        interp_type='genericLabel',
    )

    # now do a region focused registration
    synL = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=input_image_region_segmentation,
        templateHemi=input_template_region_segmentation,
        whichHemi=1,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "region_reg",
        total_sigma=total_sigma,
    )

    ants.image_write(synL['warpedmovout'], output_prefix + "region_reg.nii.gz" )

    fignameL = output_prefix + "_region_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL, black_bg=False, crop=True )

    lhjac = ants.create_jacobian_determinant_image(
        synL['warpedmovout'],
        synL['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( lhjac, output_prefix+'region_jacobian.nii.gz' )

    return {
        "synL":synL,
        "synLpng":fignameL,
        "lhjac":lhjac
        }


def t1_hypointensity( x, xsegmentation, xWMProbability, template, templateWMPrior, wmh_thresh=0.1 ):
    """
    provide measurements that may help decide if a given t1 image is likely
    to have hypointensity.

    x: input image; bias-corrected, brain-extracted and denoised

    xsegmentation: input image hard-segmentation results

    xWMProbability: input image WM probability

    template: template image

    templateWMPrior: template tissue prior

    wmh_thresh: float used to threshold WMH probability and produce summary data

    returns:

        - wmh_summary: summary data frame based on thresholding WMH probability at wmh_thresh
        - wmh_probability_image: probability image denoting WMH probability; higher values indicate
          that WMH is more likely
        - wmh_evidence_of_existence: an integral evidence that indicates the likelihood that the input
            image content supports the presence of white matter hypointensity.
            greater than zero is supportive of WMH.  the higher, the more so.
            less than zero is evidence against.
        - wmh_max_prob: max probability of wmh
        - features: the features driving WMH predictons

    """
    mybig = [88,128,128]
    templatesmall = ants.resample_image( template, mybig, use_voxels=True )
    qaff = ants.registration(
        ants.rank_intensity(x),
        ants.rank_intensity(templatesmall), 'SyN',
        syn_sampling=2,
        syn_metric='CC',
        reg_iterations = [25,15,0,0],
        aff_metric='GC', random_seed=1 )
    afftx = qaff['fwdtransforms'][1]
    templateWMPrior2x = ants.apply_transforms( x, templateWMPrior, qaff['fwdtransforms'] )
    cerebrum = ants.threshold_image( xsegmentation, 2, 4 )
    realWM = ants.threshold_image( templateWMPrior2x , 0.1, math.inf )
    inimg = ants.rank_intensity( x )
    parcellateWMdnz = ants.kmeans_segmentation( inimg, 2, realWM, mrf=0.3 )['probabilityimages'][0]
    x2template = ants.apply_transforms( templatesmall, x, afftx, whichtoinvert=[True] )
    parcellateWMdnz2template = ants.apply_transforms( templatesmall,
      cerebrum * parcellateWMdnz, afftx, whichtoinvert=[True] )
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
    # lesresam = lesresam * cerebrum
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



def deep_nbm( t1, ch13_weights, nbm_weights, registration=True,
    csfquantile = 0.15, verbose=False ):

    """
    Nucleus basalis of Meynert segmentation and subdivision

    Perform NBM segmentation in T1 images using Avants labels.

    t1 : T1-weighted neuroimage antsImage - already brain extracted

    ch13_weights : string weight file for ch13

    nbm_weights : string weight file for nbm

    registration : boolean to correct for image orientation and resolution by registration

    csfquantile : float value below 0.5 that tries to trim residual CSF off brain.

    The labeling is as follows:

    Label,Description,Side
    1,CH13_left,left
    2,CH13_right,right
    3,NBM_left_ant,left
    4,NBM_left_mid,left
    5,NBM_left_pos,left
    6,NBM_right_ant,right
    7,NBM_right_mid,right
    8,NBM_right_pos,right

    Failure modes will include odd image orientation (in which case you might
    use the registration option).  A more nefarious issue can be a poor extraction
    of the cerebrum in the inferior frontal lobe.  These can be unpredictable
    but if one sees a bad extraction, please check the mask that is output by
    this function to see if it excludes non-cerebral tissue.

    """

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    label_descriptions = ['background',
                          'CH13_left',
                          'CH13_right',
                          'NBM_left_ant',
                          'NBM_left_mid',
                          'NBM_left_pos',
                          'NBM_right_ant',
                          'NBM_right_mid',
                          'NBM_right_pos',
                          ]

    t1use = ants.iMath( t1, "Normalize" )
    if registration:
        nbmtemplate = ants.image_read(get_data("nbm_template", target_extension=".nii.gz"))
        orireg = ants.registration( fixed = nbmtemplate,
            moving = t1use,
            type_of_transform="antsRegistrationSyNQuickRepro[a]", verbose=False )
        t1use = orireg['warpedmovout']

    template = ants.image_read(get_data("CIT168_T1w_700um_pad_adni", target_extension=".nii.gz"))
    templateSmall = ants.resample_image( template, [2.0,2.0,2.0] )
    registrationsyn = ants.registration(
        fixed=templateSmall,
        moving=ants.iMath(t1use,"Normalize"),
        type_of_transform="antsRegistrationSyNQuickRepro[s]", verbose=False )

    if verbose:
        print( registrationsyn['fwdtransforms'] )

    image = ants.iMath( t1use, "TruncateIntensity", 0.0001, 0.999 ).iMath("Normalize")
    bfPriorL1 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_1_left", target_extension=".nii.gz"))
    bfPriorR1 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_1_right", target_extension=".nii.gz"))
    bfPriorL2 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_2_left", target_extension=".nii.gz"))
    bfPriorR2 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_2_right", target_extension=".nii.gz"))

    patchSize = [ 64, 64, 32 ]
    priorL1tosub = ants.apply_transforms( image, bfPriorL1, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorR1tosub = ants.apply_transforms( image, bfPriorR1, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorL2tosub = ants.apply_transforms( image, bfPriorL2, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorR2tosub = ants.apply_transforms( image, bfPriorR2, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")

    masker = ants.threshold_image(image, np.quantile(image[image>1e-4], csfquantile ), 1e9 )

    ch13point = ants.get_center_of_mass( priorL1tosub + priorR1tosub )
    def special_crop( x, pt, domainer ):
        pti = np.round( ants.transform_physical_point_to_index( x, pt ) )
        xdim = x.shape
        for k in range(len(xdim)):
            if pti[k] < 0:
                pti[k]=0
            if pti[k] > (xdim[k]-1):
                pti[k]=(xdim[k]-1)
        mim = ants.make_image( domainer )
        ptioff = pti.copy()
        for k in range(len(xdim)):
            ptioff[k] = ptioff[k] - np.round( domainer[k] / 2 )
        domainerlo = []
        domainerhi = []
        for k in range(len(xdim)):
            domainerlo.append( int(ptioff[k] - 1) )
            domainerhi.append( int(ptioff[k] + 1) )
        loi = ants.crop_indices( x, tuple(domainerlo), tuple(domainerhi) )
        mim = ants.copy_image_info(loi,mim)
        return ants.resample_image_to_target( x, mim )

    nchanCH13 = 1
    nclasstosegCH13 = 3 # for ch13
    nchanNBM = 2
    nclasstosegNBM = 4 # for nbm
    actor = 'classification'
    nfilt = 32
    addoptsNBM = "nnUnetActivationStyle"
    unetCH13 = antspynet.create_unet_model_3d(
         [ None, None, None, nchanCH13 ],
         number_of_outputs = nclasstosegCH13, # number of landmarks must be known
         number_of_layers = 4, # should optimize this wrt criterion
         number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
         convolution_kernel_size = 3, # maybe should optimize this wrt criterion
         deconvolution_kernel_size = 2,
         pool_size = 2,
         strides = 2,
         dropout_rate = 0,
         weight_decay = 0,
         mode = 'classification' )
    unetCH13.load_weights( ch13_weights )

    physspace = special_crop( image, ch13point, patchSize)
    ch13array = physspace.numpy()
    newshape = list( ch13array.shape )
    newshape.insert(0,1)
    newshape.append(1)
    ch13pred = unetCH13.predict( tf.reshape( ch13array, newshape ) )
    probability_images = []
    for jj in range(3):
        temp = ants.from_numpy( ch13pred[0,:,:,:,jj] )
        probability_images.append( ants.copy_image_info( physspace, temp ) )
    bint = physspace * 0 + 1
    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], bint )
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], bint),
        np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
    relabeled_image = ants.image_clone(segmentation_image)
    ch13totalback = ants.resample_image_to_target(relabeled_image, image, interp_type='nearestNeighbor') * masker
    if registration:
        ch13totalback = ants.apply_transforms( t1, ch13totalback,
            orireg['invtransforms'][0], whichtoinvert=[True], interpolator='nearestNeighbor' )

    if verbose:
        print("CH13 done")

    maskind = 3
    nlayers =  4 # for unet
    unet1 = antspynet.create_unet_model_3d(
         [ None, None, None, 2 ],
          number_of_outputs = 1, # number of landmarks must be known
           number_of_layers = 4, # should optimize this wrt criterion
           number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
           convolution_kernel_size = 3, # maybe should optimize this wrt criterion
           deconvolution_kernel_size = 2,
           pool_size = 2,
           strides = 2,
           dropout_rate = 0.0,
           weight_decay = 0,
           additional_options = "nnUnetActivationStyle",
           mode = "sigmoid" )
    maskinput = tf.keras.layers.Input( [ None, None,  None, 1 ] )
    posteriorMask1 = tf.keras.layers.multiply(
      [ unet1.outputs[0] , maskinput ], name='maskTimesPosteriors1'  )
    unet = tf.keras.models.Model( [ unet1.inputs[0], maskinput ], posteriorMask1 )

    unet2 = antspynet.create_unet_model_3d(
         [ None, None, None, 2 ],
          number_of_outputs = nclasstosegNBM, # number of landmarks must be known
           number_of_layers = 4, # should optimize this wrt criterion
           number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
           convolution_kernel_size = 3, # maybe should optimize this wrt criterion
           deconvolution_kernel_size = 2,
           pool_size = 2,
           strides = 2,
           dropout_rate = 0.0,
           weight_decay = 0,
           additional_options = "nnUnetActivationStyle",
           mode =  "classification" )

    temp = tf.split( unet1.inputs[0], 2, axis=4 )
    temp[1] = unet.outputs[0]
    newmult = tf.concat( temp, axis=4 )
    unetonnet = unet2( newmult )
    unetNBM = tf.keras.models.Model(
        unet.inputs,
        [ unetonnet,  unet1.outputs[0] ] )
    unetNBM.load_weights( nbm_weights )

    # do each side separately
    bfseg = t1 * 0.0
    for nbmnum in [0,1]:
        if nbmnum == 0:
            nbmpoint = ants.get_center_of_mass( priorL2tosub )
            nbmprior = special_crop( priorL2tosub, nbmpoint, patchSize).numpy() # prior
            labels=[3,4,5]
        if nbmnum == 1:
            nbmpoint = ants.get_center_of_mass( priorR2tosub )
            nbmprior = special_crop( priorR2tosub, nbmpoint, patchSize).numpy() # prior
            labels=[6,7,8]
        physspaceNBM = special_crop( image, nbmpoint, patchSize) # image
        nbmmask = special_crop( masker, nbmpoint, patchSize).numpy() # mask
        tfarr1 = tf.stack( [physspaceNBM.numpy(),nbmprior], axis=3  )
        newshapeNBM = list( tfarr1.shape )
        newshapeNBM.insert(0,1)
        tfarr1 = tf.reshape(tfarr1, newshapeNBM )
        tfarr2 = tf.reshape( nbmmask, newshape )
        nbmpred = unetNBM.predict( ( tfarr1, tfarr2  ) )
        segpred = nbmpred[0]
        sigmoidpred = nbmpred[1]
        nbmpred1_image = ants.from_numpy( sigmoidpred[0,:,:,:,0] )
        nbmpred1_image = ants.copy_image_info( physspaceNBM, nbmpred1_image )
        bint = ants.threshold_image( nbmpred1_image, 0.5, 1.0 ).iMath("GetLargestComponent")
        probability_images = []
        for jj in range(3):
            temp = ants.from_numpy( segpred[0,:,:,:,jj+1] )
            probability_images.append( ants.copy_image_info( physspaceNBM, temp ) )
        image_matrix = ants.image_list_to_matrix(probability_images, bint)
        segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1)
        segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
        relabeled_image = ants.image_clone(segmentation_image)
        for i in range(len(labels)):
            relabeled_image[segmentation_image==(i+1)] = labels[i]
        relabeled_image = ants.resample_image_to_target(relabeled_image, image, interp_type='nearestNeighbor')
        if registration:
            relabeled_image = ants.apply_transforms( t1, relabeled_image,
                    orireg['invtransforms'][0], whichtoinvert=[True],
                    interpolator='nearestNeighbor' )
        if verbose:
            print("NBM" + str( nbmnum ) )
        bfseg = bfseg + relabeled_image
    bfseg = ch13totalback + bfseg * ants.threshold_image( ch13totalback, 0, 0 )
    bfsegdesc = map_segmentation_to_dataframe( 'nbm3CH13', bfseg )

    if registration:
        masker = ants.apply_transforms( t1, masker,
            orireg['invtransforms'][0], whichtoinvert=[True],
            interpolator='nearestNeighbor' )

    return { 'segmentation':bfseg, 'description':bfsegdesc, 'mask': masker }

def hierarchical( x, output_prefix, labels_to_register=[2,3,4,5],
    imgbxt=None, cit168 = False, is_test=False, verbose=True ):
    """
    Default processing for a T1-weighted image.  See README.

    Arguments
    ---------
    x : T1-weighted neuroimage antsImage

    output_prefix: string directory and prefix

    labels_to_register: list of integer segmentation labels (of 1 to 6 as defined
    by atropos: csf, gm, wm, dgm, brainstem, cerebellum) to define
    the tissue types / regions of the brain to register.  set to None to
    skip registration which will be faster but omit some results.

    imgbxt : pre-existing brain extraction - a binary image - will disable some processing

    cit168 : boolean returns labels from CIT168 atlas

    is_test: boolean ( parameters to run more quickly but with low quality )

    verbose: boolean

    Returns
    -------
    dataframes and associated derived data

        - brain_n4_dnz : extracted brain denoised and bias corrected
        - brain_extraction : brain mask
        - rbp:  random basis projection results
        - left_right : left righ hemisphere segmentation
        - dkt_parc : dictionary object containing segmentation labels
        - registration : dictionary object containing registration results
        - hippLR : dictionary object containing hippocampus results
        - medial_temporal_lobe : dictionary object containing deep_flash (medial temporal lobe parcellation) results
        - white_matter_hypointensity : dictionary object containing WMH results
        - wm_tractsL  : white matter tracts, left
        - wm_tractsR  : white matter tracts, right
        - dataframes : summary data frames

    """
    if verbose:
        print("Read")
    tfn = get_data('T_template0', target_extension='.nii.gz' )
    tfnw = get_data('T_template0_WMP', target_extension='.nii.gz' )
    tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )
    bfn = antspynet.get_antsxnet_data( "croppedMni152" )

    ##### read images and do simple bxt ops
    templatea = ants.image_read( tfn )
    if verbose:
        print("bxt")
    templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
    templateawmprior = ants.image_read( tfnw )
    templatealr = ants.image_read( tlrfn )
    templateb = ants.image_read( bfn )
    templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
    if imgbxt is None:
        probablySR = False
        imgbxt = brain_extraction( x )
        img = x * imgbxt
    else:
        probablySR = True
        img = ants.image_clone( x )

    if verbose:
        print("rbp")

    # this is an unbiased method for identifying predictors that can be used to
    # rank / sort data into clusters, some of which may be associated
    # with outlierness or low-quality data
    templatesmall = ants.resample_image( templateb, (91,109,91), use_voxels=True )
    rbp = random_basis_projection( img, templatesmall )

    if verbose:
        print("intensity")

    ##### intensity modifications
    img = ants.iMath( img, "Normalize" ) * 255.0
    if not probablySR:
        img = ants.denoise_image( img, imgbxt, noise_model='Gaussian')
        img = ants.n4_bias_field_correction( img ).iMath("Normalize")

    # optional - quick look at result
    bxt_png = output_prefix + "_brain_extraction_dnz_n4_view.png"
    ants.plot(img,axis=2,ncol=8,nslices=24, crop=True, black_bg=False,
        filename = bxt_png )

    if verbose:
        print("hemi")

    # assuming data is reasonable quality, we should proceed with the rest ...
    mylr = label_hemispheres( img, templatea, templatealr )

    if verbose:
        print("parcellation")

    ##### hierarchical labeling
    myparc = deep_brain_parcellation( img, templateb,
        do_cortical_propagation = not is_test, verbose=False )

    ##### accumulate data into data frames
    hemi = map_segmentation_to_dataframe( "hemisphere", myparc['hemisphere_labels'] )
    tissue = map_segmentation_to_dataframe( "tissues", myparc['tissue_segmentation'] )
    dktl = map_segmentation_to_dataframe( "lobes", myparc['dkt_lobes'] )
    dktp = map_segmentation_to_dataframe( "dkt", myparc['dkt_parcellation'] )
    dktc = None
    if not is_test:
        dktc = map_segmentation_to_dataframe( "dkt", myparc['dkt_cortex'] )

    tissue_seg_png = output_prefix + "_seg.png"
    ants.plot( img, myparc['tissue_segmentation'], axis=2, nslices=21, ncol=7,
        alpha=0.6, filename=tissue_seg_png,
        crop=True, black_bg=False )

    if verbose:
        print("WMH")

    ##### below here are more exploratory nice to have outputs
    myhypo = t1_hypointensity(
        img,
        myparc['tissue_segmentation'], # segmentation
        myparc['tissue_probabilities'][3], # wm posteriors
        templatea,
        templateawmprior )

    if verbose:
        print("registration")

    ##### traditional deformable registration as a high-resolution complement to above
    wm_tractsL = None
    wm_tractsR = None
    wmtdfL = None
    wmtdfR = None
    reg = None
    if labels_to_register is not None:
        reg = hemi_reg(
            input_image = img,
            input_image_tissue_segmentation = myparc['tissue_segmentation'],
            input_image_hemisphere_segmentation = mylr,
            input_template=templatea,
            input_template_hemisphere_labels=templatealr,
            output_prefix = output_prefix + "_SYN",
            labels_to_register = labels_to_register,
            is_test=is_test )
        if verbose:
            print("wm tracts")
        ##### how to use the hemi-reg output to generate any roi value from a template roi
        wm_tracts = ants.image_read( get_data( "wm_major_tracts", target_extension='.nii.gz' ) )
        wm_tractsL = ants.apply_transforms( img, wm_tracts, reg['synL']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( mylr, 1, 1  )
        wm_tractsR = ants.apply_transforms( img, wm_tracts, reg['synR']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( mylr, 2, 2  )
        wmtdfL = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsL )
        wmtdfR = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsR )

    cit168lab = None
    cit168reg = None
    cit168lab_desc = None
    if cit168:
        if verbose:
            print("cit168")
        cit168adni = get_data( "CIT168_T1w_700um_pad_adni",target_extension='.nii.gz')
        cit168adni = ants.image_read( cit168adni ).iMath("Normalize")
        cit168reg = region_reg(
            input_image = img,
            input_image_tissue_segmentation=myparc['tissue_segmentation'],
            input_image_region_segmentation=imgbxt,
            input_template=cit168adni,
            input_template_region_segmentation=ants.threshold_image( cit168adni, 0.15, 1 ),
            output_prefix=output_prefix + "_CIT168RRSYN",
            padding=10,
            labels_to_register = [1,2,3,4,5,6],
            total_sigma=0.1,
            is_test=is_test )['synL']
        cit168lab = get_data( "det_atlas_25_pad_LR_adni", target_extension='.nii.gz' )
        cit168lab = ants.image_read( cit168lab )
        cit168lab = ants.apply_transforms( img, cit168lab,
                cit168reg['invtransforms'], interpolator = 'genericLabel' )
        cit168lab_desc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', cit168lab )

    if verbose:
        print("hippocampus")

    ##### specialized labeling for hippocampus
    ntries = 10
    if is_test:
        ntries = 1
    hippLR = deep_hippo( img, templateb, ntries )

    if verbose:
        print("medial temporal lobe")

    ##### deep_flash medial temporal lobe parcellation
    deep_flash = deep_mtl(img)

    if verbose:
        print("NBM")

    ##### deep_nbm basal forebrain parcellation
    deep_bf = deep_nbm( img,
        get_data("ch13_weights",target_extension='.h5'),
        get_data("nbm3_weights",target_extension='.h5') )

    mydataframes = {
        "hemispheres":hemi,
        "tissues":tissue,
        "dktlobes":dktl,
        "dktregions":dktp,
        "dktcortex":dktc,
        "wmtracts_left":wmtdfL,
        "wmtracts_right":wmtdfR,
        "wmh":myhypo['wmh_summary'],
        "mtl":deep_flash['mtl_description'],
        "bf":deep_bf['description'],
        "cit168":cit168lab_desc
        }

    outputs = {
        "brain_n4_dnz": img,
        "brain_n4_dnz_png": bxt_png,
        "brain_extraction": imgbxt,
        "tissue_seg_png": tissue_seg_png,
        "rbp": rbp,
        "left_right": mylr,
        "dkt_parc": myparc,
        "registration":reg,
        "hippLR":hippLR,
        "medial_temporal_lobe":deep_flash,
        "white_matter_hypointensity":myhypo,
        "wm_tractsL":wm_tractsL,
        "wm_tractsR":wm_tractsR,
        "mtl":deep_flash['mtl_segmentation'],
        "bf":deep_bf['segmentation'],
        "cit168lab":  cit168lab,
        "cit168reg":  cit168reg,
        "dataframes": mydataframes
    }

    return outputs


def trim_segmentation_by_distance( segmentation, which_label, distance ):
    """
    trim a segmentation by the distance provided by the user. computes a distance
    transform from the segmentation - treated as binary - and trims the target
    label by that distance.

    Arguments
    ---------
    segmentation : ants image segmentation

    which_label : the label to trim

    distance : float distance value

    Returns
    -------
    trimmed_segmentation

    Example
    -------
    >>> import ants
    >>> img = ants.image_read( ants.get_data( 'r16' ) )
    >>> seg = ants.threshold_image( img, "Otsu", 3 )
    >>> tseg = antspyt1w.trim_segmentation_by_distance( seg, 1, 10 )
    """
    bseg = ants.threshold_image( segmentation, 1, segmentation.max() )
    dist = ants.iMath( bseg, "MaurerDistance" ) * (-1.0)
    disttrim = ants.threshold_image( dist, distance, dist.max() )
    tarseg = ants.threshold_image( segmentation, which_label, which_label ) * disttrim
    segmentationtrim = segmentation.clone()
    segmentationtrim[ segmentation == which_label ] = 0
    return segmentationtrim + tarseg * which_label



def zoom_syn( target_image, template, template_segmentations,
    initial_registration,
    dilation = 4,
    regIterations = [25] ):
    """
    zoomed in syn - a hierarchical registration applied to a hierarchical segmentation

    Initial registration is followed up by a refined and focused high-resolution registration.
    This is performed on the cropped image where the cropping region is determined
    by the first segmentation in the template_segmentations list.  Segmentations
    after the first one are assumed to exist as sub-regions of the first.  All
    segmentations are assumed to be binary.

    Arguments
    ---------
    target_image : ants image at original resolution

    template : ants image template to be mapped to the target image

    template_segmentations : list of binary segmentation images

    dilation : morphological dilation amount applied to the first segmentation and used for cropping

    regIterations : parameter passed to ants.registration

    Returns
    -------
    dictionary
        containing segmentation and registration results in addition to cropping results

    Example
    -------
    >>> import ants
    >>> ireg = ants.registration( target_image, template, "antsRegistrationSyNQuickRepro[s]" )
    >>> xxx = antspyt1w.zoom_syn(  orb,  template, level2segs, ireg )
    """
    croppertem = ants.iMath( template_segmentations[0], "MD", dilation )
    templatecrop = ants.crop_image( template, croppertem )
    cropper = ants.apply_transforms( target_image,
        croppertem, initial_registration['fwdtransforms'],
        interpolator='linear' ).threshold_image(0.5,1.e9)
    croplow = ants.crop_image( target_image,  cropper )
    synnerlow = ants.registration( croplow, templatecrop,
        'SyNOnly', gradStep = 0.20, regIterations = regIterations, randomSeed=1,
        initialTransform = initial_registration['fwdtransforms'] )
    orlist = []
    for jj in range(len(template_segmentations)):
      target_imageg = ants.apply_transforms( target_image, template_segmentations[jj],
        synnerlow['fwdtransforms'],
        interpolator='linear' ).threshold_image(0.5,1e9)
      orlist.append( target_imageg )
    return{
          'segmentations': orlist,
          'registration': synnerlow,
          'croppedimage': croplow,
          'croppingmask': cropper
          }
