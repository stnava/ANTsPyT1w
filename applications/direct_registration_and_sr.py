# this script assumes the image have been brain extracted
import os.path
from os import path

# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import ljlf_parcellation_one_template
from superiq import list_to_string
# from pipeline_utils import *


# user definitions here
tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
tfn = tdir + "CIT168_T1w_700um_pad.nii.gz"
tfnl = tdir + "det_atlas_25_pad_LR.nii.gz"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
# config handling
output_filename = "outputs/ZZZ_"
# input data
imgIn = ants.image_read( infn )
# imgIn = ants.denoise_image( imgIn, noise_model='Rician' )
imgIn = ants.iMath( imgIn, "TruncateIntensity", 0.00001, 0.9995 ).iMath("Normalize")

# brain age
if False:
    t1_preprocessing = antspynet.preprocess_brain_image( imgIn,
            truncate_intensity=(0.00001, 0.9995),
            do_brain_extraction=False,
            template="croppedMni152",
            template_transform_type="AffineFast",
            do_bias_correction=False,
            do_denoising=False,
            antsxnet_cache_directory="/tmp/",
            verbose=True)
    bage = antspynet.brain_age( t1_preprocessing['preprocessed_image'], do_preprocessing=False )
# save these values to a csv file
template = ants.image_read(tfn)
templateL = ants.image_read(tfnl)
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

# expected output data
output_filename_jac = output_filename + "_jacobian.nii.gz"
output_filename_seg = output_filename + "_ORseg.nii.gz"
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_segljlf = output_filename  +  "_SR_segljlf.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
output_filename_warped = output_filename  + "_warped.nii.gz"

is_test=True

regsegits=[200,200,200,50]

if is_test:
    regsegits=[200,200,200,10]

if not 'reg' in locals():
    regits = [600,600,600,200,50]
    lregits = [100, 100, 100, 55]
    verber=False
    if is_test:
        regits=[600,600,0,0,0]
        lregits=[600,60,0,0,0]
        verber=True
    reg = ants.registration( template, imgIn,
#        type_of_transform="TV[2]",        grad_step = 1.4,
        type_of_transform="SyN",        grad_step = 0.20,
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=regits, verbose=verber )
    print("SyN Done")

ants.image_write( reg['warpedmovout'], output_filename_warped )
myjacobian=ants.create_jacobian_determinant_image( template, reg['fwdtransforms'][0], True )
ants.image_write( myjacobian, output_filename_jac )
inv_transforms = reg['invtransforms']
initlab0 = ants.apply_transforms( imgIn, templateL, inv_transforms, interpolator="nearestNeighbor" )
ants.image_write( initlab0, output_filename_seg )
g1 = ants.label_geometry_measures(initlab0,imgIn)
sr_params = { 'upFactor':[2,2,2], 'dilation_amount':8, 'verbose':False}
mynums=list( range(1,32) )

if is_test:
    sr_params = { 'upFactor':[2,2,2], 'dilation_amount':2, 'verbose':True}
    mynums=[1,2,5,6,7,8,9,10,17,18,21,22,23,24,25,26]

if not 'srseg' in locals():
    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = mynums,
        dilation_amount = sr_params['dilation_amount'],
        max_lab_plus_one = True,
        verbose = sr_params['verbose']
    )

if not 'ljlfseg' in locals():
    ljlfseg = ljlf_parcellation_one_template(
            img = srseg['super_resolution'],
            segmentation_numbers = mynums,
            forward_transforms = inv_transforms,
            template = template,
            templateLabels = templateL,
            templateRepeats = 8,
            submask_dilation = 6,
            searcher=1,
            radder=2,
            reg_iterations=lregits,
            syn_sampling=2,
            syn_metric='CC',
            max_lab_plus_one=True,
            deformation_sd=2.0,
            intensity_sd=0.1,
            output_prefix=output_filename,
            verbose=False,
        )


ants.image_write( srseg['super_resolution'], output_filename_sr )

ants.image_write( srseg['super_resolution_segmentation'], output_filename_sr_seg )

ants.image_write( ljlfseg['segmentation'], output_filename_sr_segljlf )

# below - do a good registration for each label in order to get a locally
# high quality registration - and also to get the jacobian
localregsegtotal = srseg['super_resolution'] * 0.0
if True:
    label_geo_list = []
    label_groups = []
    label_groups.append( [1,5,6] )       # r putamen & gp
    label_groups.append( [17,21,22] )    # l putamen & gp
    label_groups.append( [2] )           # l caud
    label_groups.append( [18] )          # r caud
    label_groups.append( [7,8,9,10] )    # l SN+
    label_groups.append( [23,24,25,26] ) # r SN+
    for mylab in label_groups:
        localprefix = output_filename + "synlocal_label" + list_to_string( mylab ) + "_"
        print(localprefix)
        cmskt = ants.mask_image( templateL, templateL, mylab, binarize=True ).iMath( "MD", 8 )
        cimgt = ants.crop_image( template, cmskt ) \
            .resample_image( [0.5,0.5,0.5],use_voxels=False, interp_type=0 )
        print(cimgt)
        cmsk = ants.mask_image(
            ljlfseg['segmentation'],
            ljlfseg['segmentation'],
            mylab,
            binarize=True,
        ).iMath( "MD", 12 )
        cimg = ants.crop_image( srseg['super_resolution'], cmsk )
        rig = ants.registration( ants.crop_image( cmskt ) , cmsk, "Rigid" )
        syn = ants.registration(
            ants.iMath(cimgt,"Normalize"),
            srseg['super_resolution'],
            type_of_transform="SyNOnly",
            reg_iterations=regsegits,
            initial_transform=rig['fwdtransforms'][0],
            syn_metric='cc',
            syn_sampling=2,
            outprefix=localprefix,
            verbose=False,
        )
        if len( syn['fwdtransforms'] ) > 1 :
            jimg = ants.create_jacobian_determinant_image(
                cimgt,
                syn['fwdtransforms'][0],
                True,
                False,
            )
            ants.image_write( jimg, localprefix + "jacobian.nii.gz" )
            cmskt = ants.mask_image( templateL, templateL, mylab, binarize=False )
            localregseg = ants.apply_transforms(
                srseg['super_resolution'],
                cmskt,
                syn['invtransforms'],
                interpolator='genericLabel'
            )
            localgeo = ants.label_geometry_measures(
                localregseg,
                localregseg,
            )
            output_filename_sr_regseg_csv = localprefix  + "SR_regseg.csv"
            localgeo.to_csv(output_filename_sr_regseg_csv )
            label_geo_list.append( localgeo )
            ants.image_write( syn['warpedmovout'], localprefix + "_localreg.nii.gz" )
            ants.image_write( localregseg, localprefix + "_localregseg.nii.gz" )
            # this is a hack fix to get rid of multiple labels overlapping
            # should use the usual voting scheme or just rely on the local labels
            # the latter are appropriate for shape analysis in the future.
            localregseg = localregseg * ants.threshold_image(localregsegtotal,0,0)
            localregsegtotal = localregseg + localregsegtotal

localprefix = output_filename + "_synlocal_regseg.nii.gz"
ants.image_write( localregsegtotal, localprefix )
