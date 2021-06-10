# this script assumes the image have been brain extracted
import os
import sys
import ia_batch_utils as batch

def import_handling(config):
    try:
        threads = os.environ['cpu_threads']
    except KeyError:
        threads = "8"
    # set number of threads - this should be optimized per compute instance
    os.environ["TF_NUM_INTEROP_THREADS"] = threads
    os.environ["TF_NUM_INTRAOP_THREADS"] = threads

    if config.ants_random_seed != '-1':
        os.environ['ANTS_RANDOM_SEED'] = config.ants_random_seed

    if config.itk_threads != '-1':
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = config.itk_threads
    else:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

    global ants, antspynet, pd, np, superiq

    import ants
    import antspynet
    import pandas as pd
    import numpy as np
    import superiq

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
def localsyn(img, template, hemiS, templateHemi, whichHemi, tbftotLoc, ibftotLoc, padder, iterations ):
    ihemi=img*ants.threshold_image( hemiS, whichHemi, whichHemi )
    themi=template*ants.threshold_image( templateHemi, whichHemi, whichHemi )
    rig = ants.registration( tbftotLoc, ibftotLoc, 'Affine', random_seed = 1  )
    tbftotLoct = ants.threshold_image( tbftotLoc, 0.25, 2.0 ).iMath("MD", padder )
    tcrop = ants.crop_image( themi, tbftotLoct )
    syn = ants.registration( tcrop, ihemi, 'SyNOnly',
        syn_metric='CC', syn_sampling=2, reg_iterations=iterations,
        initial_transform=rig['fwdtransforms'][0], verbose=False, random_seed = 1 )
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
    img=ants.image_read(img_path)

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


    # now do a BF focused registration

    ibftotL = bfprob1L + bfprob2L
    tbftotL = (templateBF[0]+ templateBF[2]) \
        .resample_image_to_target( template, interp_type='linear')
    ibftotR = bfprob1R + bfprob2R
    tbftotR = (templateBF[1] + templateBF[3]) \
        .resample_image_to_target( template, interp_type='linear')
    synL = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=hemiS,
        templateHemi=templateHemi,
        whichHemi=1,
        tbftotLoc=tbftotL,
        ibftotLoc=ibftotL,
        padder=6,
        iterations=regsegits,
    )
    synR = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=hemiS,
        templateHemi=templateHemi,
        whichHemi=2,
        tbftotLoc=tbftotR,
        ibftotLoc=ibftotR,
        padder=6,
        iterations=regsegits,
    )
    bftoiL1 = ants.apply_transforms(
        img,
        ants.resample_image_to_target(templateBF[0], template, interp_type='linear'),
        synL['invtransforms'],
    )
    bftoiL2 = ants.apply_transforms(
        img,
        ants.resample_image_to_target(templateBF[2], template, interp_type='linear'),
        synL['invtransforms'],
    )
    bftoiR1 = ants.apply_transforms(
        img,
        ants.resample_image_to_target(templateBF[1], template, interp_type='linear'),
        synR['invtransforms'],
    )
    bftoiR2 = ants.apply_transforms(
        img,
        ants.resample_image_to_target(templateBF[3], template, interp_type='linear'),
        synR['invtransforms'],
    )

    # get the volumes for each region (thresholded) and its sum
    myspc = ants.get_spacing( img )
    vbfL1 = np.asarray(myspc).prod() * bftoiL1.sum()
    vbfL2 = np.asarray(myspc).prod() * bftoiL2.sum()
    vbfR1 = np.asarray(myspc).prod() * bftoiR1.sum()
    vbfR2 = np.asarray(myspc).prod() * bftoiR2.sum()

    # same calculation but explicitly restricted to brain tissue
    onlygm = ants.threshold_image( idap, 2, 4 )
    vbfL1t = np.asarray(myspc).prod() * (bftoiL1*onlygm).sum()
    vbfL2t = np.asarray(myspc).prod() * (bftoiL2*onlygm).sum()
    vbfR1t = np.asarray(myspc).prod() * (bftoiR1*onlygm).sum()
    vbfR2t = np.asarray(myspc).prod() * (bftoiR2*onlygm).sum()

    volumes = {
        f"BFLCH13_{c.resolution}": vbfL1,
        f"BFLNBM_{c.resolution}": vbfL2,
        f"BFRCH13_{c.resolution}": vbfR1,
        f"BFRNBM_{c.resolution}": vbfR2,
        f"BFLCH13tissue_{c.resolution}": vbfL1t,
        f"BFLNBMtissue_{c.resolution}": vbfL2t,
        f"BFRCH13tissue_{c.resolution}": vbfR1t,
        f"BFRNBMtissue_{c.resolution}": vbfR2t,
    }
    output = c.output_file_prefix
    if not os.path.exists(output):
        os.makedirs(output)

    if c.resolution == 'OR':
        model = batch.get_s3_object(c.model_bucket, c.model_prefix, tdir)
        plist = [bftoiL1,bftoiR1,bftoiL2,bftoiR2]
        ss = superiq.super_resolution_segmentation_with_probabilities(img,plist,model)

        sr_images = ss['sr_intensities']
        sr_probs = ss['sr_probabilities']
        labels = ['BFLCH13_SRWP', 'BFRCH13_SRWP', 'BFLNBM_SRWP', 'BFRNBM_SRWP']

        for i in range(len(sr_images)):
            spc = ants.get_spacing(sr_images[i])
            srvol = np.asarray(spc).prod() * sr_probs[i].sum()
            volumes[labels[i]] = srvol


    split = c.input_value.split('/')[-1].split('-')
    rec = {}
    rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
    rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
    rec['batchid'] = c.batch_id
    rec['project'] = split[0]
    rec['subject'] = split[1]
    rec['date'] = split[2]
    rec['modality'] = split[3]
    rec['repeat'] = split[4]
    rec['process'] = c.process_name
    rec['name'] = "bf_star"
    rec['version'] = c.version
    rec['extension'] = ".nii.gz"
    rec['resolution'] = c.resolution
    for k, v in volumes.items():
        rec['data'] = {}
        rec['data']['label'] = 0
        rec['data']['key'] = k
        rec['data']['value'] = v
        print(rec)
        batch.write_to_dynamo(rec)

    df = pd.DataFrame(volumes, index=[0])
    df.to_csv(output + f'_{c.resolution}_bfvolumes.csv')
    ants.image_write( bftoiL1, output+f'bfprobCH13left{c.resolution}.nii.gz' )
    ants.image_write( bftoiR1, output+f'bfprobCH13right{c.resolution}.nii.gz' )
    ants.image_write( bftoiL2, output+f'bfprobNBMleft{c.resolution}.nii.gz' )
    ants.image_write( bftoiR2, output+f'bfprobNBMright{c.resolution}.nii.gz' )

    batch.handle_outputs(
        c.output_bucket,
        c.output_prefix,
        c.input_value,
        c.process_name,
        c.version,
    )


if __name__=="__main__":
    config = sys.argv[1]
    config = batch.LoadConfig(config)
    import_handling(config)
    main(config)
