import os
import sys
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



def main(config):
    c = config
    # input brain extracted target image
    ifn_bucket = c.input_bucket
    ifn_key = c.input_value
    data = 'data/'
    ifn = batch.get_s3_object(ifn_bucket, ifn_key, data)
    img = ants.image_read(ifn)

    if not os.path.exists(c.output_file_prefix):
        os.makedirs(c.output_file_prefix)

    model_bucket = c.model_bucket
    model_key = c.model_key
    model_path = batch.get_s3_object(model_bucket, model_key, data)
    mdl = tf.keras.models.load_model( model_path ) # FIXME - parameterize this

    template_bucket = c.template_bucket
    templatefn = batch.get_s3_object(template_bucket, c.template_base, data)
    templateBF1L  =batch.get_s3_object(template_bucket,  c.templateBF1L, data)
    templateBF2L  =batch.get_s3_object(template_bucket,  c.templateBF2L, data)
    templateBF1R  =batch.get_s3_object(template_bucket,  c.templateBF1R, data)
    templateBF2R  =batch.get_s3_object(template_bucket,  c.templateBF2R, data)
    templateCIT =batch.get_s3_object(template_bucket,  c.templateCIT, data)
    templateHemi=batch.get_s3_object(template_bucket,  c.templateHemi, data)

    template = ants.image_read( templatefn )
    prefix=c.output_file_prefix

    tdap = dap( template )
    idap = dap( img )
    maskinds=[2,3,4,5]

    # if a registration to template is already computed ( which it should be ) then
    # just apply the resulting transformations otherwise compute a new reg
    # existing registrations look like prefixWarp.nii.gz, prefixGenericAffine.mat
    imgcerebrum = ants.mask_image(idap,idap,maskinds,binarize=True).iMath("GetLargestComponent")
    temcerebrum = ants.mask_image(tdap,tdap,maskinds,binarize=True).iMath("GetLargestComponent")

    #regsegits=[200,200,200,50]

    #if c.environment=='dev':
    #    regsegits=[200,200,200,10]

    if c.do_reg:
        regits = [600,600,600,200,50]
        #lregits = [100, 100, 100, 55]
        verber=True
        if c.environment=='dev':
            regits=[600,600,0,0,0]
            #lregits=[600,60,0,0,0]
            verber=True
        reg = ants.registration(
            template * temcerebrum,
            img * imgcerebrum,
            type_of_transform="SyN",
            grad_step = 0.20,
            syn_metric='CC',
            syn_sampling=2,
            reg_iterations=regits,
            outprefix=prefix,
            verbose=verber )

    # 1 is left, 2 is right
    templateHemi = ants.image_read( templateHemi )
    hemiS = ants.apply_transforms(
        img,
        templateHemi,
        reg['invtransforms'],
        interpolator='genericLabel',
    )
    # these are the standard CIT labels
    templateCIT = ants.image_read( templateCIT )
    citS = ants.apply_transforms(
        img,
        templateCIT,
        reg['invtransforms'],
        interpolator='genericLabel',
    )

    # these are the new BF labels
    bfprobs=[]
    bftot = img * 0.0
    templateBF = [templateBF1L, templateBF1R, templateBF2L,  templateBF2R]
    for x in templateBF:
        bfloc = ants.image_read(  x )
        bfloc = ants.apply_transforms( img, bfloc, reg['invtransforms'], interpolator='linear' )
        bftot = bftot + bfloc
        bfprobs.append( bfloc )

    # the segmentation is gained by thresholding each BF prob at 0.25 or thereabouts
    # and multiplying by imgcerebrum

    # write out at OR:
    ants.image_write( img*imgcerebrum, prefix+'cerebrum.nii.gz' )
    ants.image_write( idap, prefix+'tissueSegmentation.nii.gz' )
    ants.image_write( hemiS, prefix+'hemisphere.nii.gz' )
    ants.image_write( citS, prefix+'CIT168Labels.nii.gz' )
    ants.image_write( bfprobs[0], prefix+'bfprob1left.nii.gz' )
    ants.image_write( bfprobs[1], prefix+'bfprob1right.nii.gz' )
    ants.image_write( bfprobs[2], prefix+'bfprob2left.nii.gz' )
    ants.image_write( bfprobs[3], prefix+'bfprob2right.nii.gz' )

    cort_labs = ants.label_geometry_measures(idap)
    cort_labs = cort_labs[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]
    cort_labs_records = cort_labs.to_dict('records')

    split = c.input_value.split('/')[-1].split('-')
    rec = {}
    rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
    rec['batchid'] = c.batch_id
    rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
    rec['project'] = split[0]
    rec['subject'] = split[1]
    rec['date'] = split[2]
    rec['modality'] = split[3]
    rec['repeat'] = split[4]
    rec['process'] = 'hemi_sr'
    rec['version'] = c.version
    rec['name'] = "tissueSegmentation"
    rec['extension'] = ".nii.gz"
    rec['resolution'] = "OR"



    cortLR = ants.threshold_image( idap, 2, 2 ) * hemiS

    # get SNR of WM
    wmseg = ants.threshold_image( idap, 3, 3 )
    wmMean = img[ wmseg == 1 ].mean()
    wmStd = img[ wmseg == 1 ].std()
    # get SNR wrt CSF
    csfseg = ants.threshold_image( idap, 1, 1 )
    csfStd = img[ csfseg == 1 ].std()

    # this will give us super-resolution over the whole image and also SR cortex
    # however, it may be better to just re-run the seg on the output SR image
    # this took around 100GB RAM - could also do per lobe, less RAM but more time (probably)
    srseg = super_resolution_segmentation_per_label(
        img,
        cortLR,
        [2,2,2],
        sr_model=mdl,
        segmentation_numbers=[1,2],
        dilation_amount=2
    )
    idapSR = dap( srseg['super_resolution'] )
    wmsegSR = ants.threshold_image( idapSR, 3, 3 )
    wmMeanSR = srseg['super_resolution'][ wmsegSR == 1 ].mean()
    wmStdSR = srseg['super_resolution'][ wmsegSR == 1 ].std()
    csfsegSR = ants.threshold_image( idapSR, 1, 1 )
    csfStdSR = srseg['super_resolution'][ csfsegSR == 1 ].std()
    wmSNR = wmMean/wmStd
    wmcsfSNR = wmMean/csfStd
    wmSNRSR = wmMeanSR/wmStdSR
    wmcsfSNRSR = wmMeanSR/csfStdSR
    snrdf = {
        "Label": 0,
        'WMSNROR': wmSNR,
        'WMCSFSNROR': wmcsfSNR,
        'WMSNRSR': wmSNRSR,
        'WMCSFSNRSR': wmcsfSNRSR,
    }
    cort_labs_records.append(snrdf)
    for r in cort_labs_records:
        label = r['Label']
        r.pop('Label', None)
        for k,v in r.items():
            data_field = {
                "label": label,
                'key': k,
                "value": v,
            }
            rec['data'] = data_field
            batch.write_to_dynamo(rec)



    #df = pd.DataFrame(snrdf, index=[0])
    #df.to_csv(prefix + 'wmsnr.csv', index=False)
    ants.image_write( srseg['super_resolution_segmentation'], prefix+'corticalSegSR.nii.gz' )
    ants.image_write( srseg['super_resolution'], prefix+'SR.nii.gz' )
    batch.handle_outputs(
        c.output_bucket,
        c.output_prefix,
        c.input_value,
        c.process_name,
        c.version,
    )

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

    import ants
    import antspynet
    import tensorflow as tf
    import pandas as pd

    from superiq import super_resolution_segmentation_per_label
    from superiq import list_to_string

if __name__ == "__main__":
    config = sys.argv[1]
    config = batch.LoadConfig(config)
    import_handling(config)
    main(config)
