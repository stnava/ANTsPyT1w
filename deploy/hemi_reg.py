# this script assumes the image have been brain extracted
import os.path
from os import path
try:
    threads = os.environ['cpu_threads']
except KeyError:
    threads = "8"
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string
import superiq
import ia_batch_utils as batch

def dap( x ):
    bbt = ants.image_read( antspynet.get_antsxnet_data( "biobank" ) )
    bbt = antspynet.brain_extraction( bbt, "t1v0" ) * bbt
    qaff=ants.registration( bbt, x, "AffineFast" )
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
    hemicropmask = ants.threshold_image( templateHemi, whichHemi, whichHemi ).iMath("MD",padder)
    tcrop = ants.crop_image( themi, hemicropmask )
    syn = ants.registration( tcrop, ihemi, 'SyN',
        syn_metric='CC', syn_sampling=2, reg_iterations=iterations,
        verbose=False, outprefix = output_prefix )
    return syn

def find_in_list(list_, find):
    x = [i for i in list_ if i.endswith(find)]
    if len(x) != 1:
        raise ValueError(f"Find in list: {find} not in list")
    else:
        return x[0]

def main(input_config):
    c = input_config

    tdir = "data/"
    img_path = batch.get_s3_object(c.input_bucket, c.input_value, tdir)
    img=ants.image_read(img_path).iMath("Normalize")

    filter_vals = c.input_value.split('/')
    x = '/'.join(filter_vals[2:7])
    pipeline_objects = batch.list_objects(c.hemi_bucket, c.hemi_prefix + x + f'/hemi_sr/' )
    tSeg = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('tissueSegmentation.nii.gz'),
        find_in_list(pipeline_objects, "tissueSegmentation.nii.gz"),
        tdir,
    )
    hemi = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('hemisphere.nii.gz'),
        find_in_list(pipeline_objects, "hemisphere.nii.gz"),
        tdir,
    )
    citL = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('CIT168Labels.nii.gz'),
        find_in_list(pipeline_objects, "CIT168Labels.nii.gz"),
        tdir,
    )
    bfL1 = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('bfprob1left.nii.gz'),
        find_in_list(pipeline_objects, "bfprob1left.nii.gz"),
        tdir,
    )
    bfL2 = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('bfprob2left.nii.gz'),
        find_in_list(pipeline_objects, "bfprob2left.nii.gz"),
        tdir,
    )
    bfR1 = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('bfprob1right.nii.gz'),
        find_in_list(pipeline_objects, "bfprob1right.nii.gz"),
        tdir,
    )
    bfR2 = batch.get_s3_object(
        c.hemi_bucket,
        #pipeline_objects.endswith('bfprob2right.nii.gz'),
        find_in_list(pipeline_objects, "bfprob2right.nii.gz"),
        tdir,
    )

    idap=ants.image_read(tSeg).resample_image_to_target( img, interp_type='genericLabel')
    ionlycerebrum = ants.threshold_image( idap, 2, 4 )
    hemiS=ants.image_read(hemi).resample_image_to_target( img, interp_type='genericLabel')
    citS=ants.image_read(citL).resample_image_to_target( img, interp_type='genericLabel')
    bfprob1L=ants.image_read(bfL1).resample_image_to_target( img, interp_type='linear')
    bfprob1R=ants.image_read(bfR1).resample_image_to_target( img, interp_type='linear')
    bfprob2L=ants.image_read(bfL2).resample_image_to_target( img, interp_type='linear')
    bfprob2R=ants.image_read(bfR2).resample_image_to_target( img, interp_type='linear')

    template_bucket = c.template_bucket
    template = ants.image_read(batch.get_s3_object(template_bucket, c.template_base, tdir))
    templateBF1L = ants.image_read(batch.get_s3_object(template_bucket,  c.templateBF1L, tdir))
    templateBF2L = ants.image_read(batch.get_s3_object(template_bucket,  c.templateBF2L, tdir))
    templateBF1R = ants.image_read(batch.get_s3_object(template_bucket,  c.templateBF1R, tdir))
    templateBF2R = ants.image_read(batch.get_s3_object(template_bucket,  c.templateBF2R, tdir))
    templateCIT = ants.image_read(batch.get_s3_object(template_bucket,  c.templateCIT, tdir))
    templateHemi= ants.image_read(batch.get_s3_object(template_bucket,  c.templateHemi, tdir))
    templateBF = [templateBF1L, templateBF1R, templateBF2L,  templateBF2R]

    # FIXME - this should be a "good" registration like we use in direct reg seg
    # ideally, we would compute this separately - but also note that
    regsegits=[200,200,200]



    # upsample the template if we are passing SR as input
    if min(ants.get_spacing(img)) < 0.8:
        regsegits=[200,200,200,200]
        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )

    templateCIT = ants.resample_image_to_target(
        templateCIT,
        template,
        interp_type='genericLabel',
    )
    templateHemi = ants.resample_image_to_target(
        templateHemi,
        template,
        interp_type='genericLabel',
    )

    tdap = dap( template )
    tonlycerebrum = ants.threshold_image( tdap, 2, 4 )
    maskinds=[2,3,4,5]
    temcerebrum = ants.mask_image(tdap,tdap,maskinds,binarize=True).iMath("GetLargestComponent")


    output = c.output_file_prefix
    if not os.path.exists(output):
        os.makedirs(output)

    # now do a hemisphere focused registration
    mypad = 10 # pad the hemi mask for cropping - important due to diff_0
    synL = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=hemiS,
        templateHemi=templateHemi,
        whichHemi=1,
        padder=mypad,
        iterations=regsegits,
        output_prefix = output + "left_hemi_reg",
    )
    synR = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=hemiS,
        templateHemi=templateHemi,
        whichHemi=2,
        padder=mypad,
        iterations=regsegits,
        output_prefix = output + "right_hemi_reg",
    )

    fignameL = output + "left_hemi_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL)

    fignameR = output + "right_hemi_reg.png"
    ants.plot(synR['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameR)

    temp = ants.image_read(synL['fwdtransforms'][0]).split_channels()[0]
    lhjac = ants.create_jacobian_determinant_image(
        temp,
        synL['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( lhjac, output+'left_hemi_jacobian.nii.gz' )

    temp = ants.image_read(synR['fwdtransforms'][0]).split_channels()[0]
    rhjac = ants.create_jacobian_determinant_image(
        temp,
        synR['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( rhjac, output+'right_hemi_jacobian.nii.gz' )

    batch.handle_outputs(
        c.output_bucket,
        c.output_prefix,
        c.input_value,
        c.process_name,
        c.version
    )


if __name__=="__main__":
    config = sys.argv[1]
    config = batch.LoadConfig(config)
    main(config)
