import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import math
import ants
import sys
from superiq.pipeline_utils import *
from superiq import native_to_superres_ljlf_segmentation
from superiq import check_for_labels_in_image
from superiq import super_resolution_segmentation_per_label

def local_jlf(input_config):
    config = LoadConfig(input_config)
    if config.environment == "prod":
        input_image = get_pipeline_data(
            config.brain_extraction_suffix, # TODO:Possible breaking change here
            config.input_value,
            config.pipeline_bucket,
            config.pipeline_prefix,
        )

        atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
        brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
        brains.sort()
        brains = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
        brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
        brainsSeg.sort()
        brainsSeg = [ants.image_read(i) for i in brainsSeg]
    elif config.environment == 'val':
        input_image = get_s3_object(config.input_bucket, config.input_value, "data")
        target_image_base = input_image.split('/')[-1].split('.')[0]
        target_image_label_name = \
            config.atlas_label_prefix +  target_image_base + '_JLFSegOR.nii.gz'
        target_image_labels_path = get_s3_object(
            config.input_bucket,
            target_image_label_name,
            "data",
        )

        atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
        atlas_image_keys = [i for i in atlas_image_keys if i != input_image]
        brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
        brains.sort()
        brains = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
        atlas_label_keys = [i for i in atlas_label_keys if i != target_image_labels_path]
        brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
        brainsSeg.sort()
        brainsSeg = [ants.image_read(i) for i in brainsSeg]
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")
    input_image = ants.image_read(input_image)

    # Noise Correction
    input_image = ants.denoise_image(input_image)
    input_image = ants.iMath(input_image, 'TruncateIntensity', 0.000001, 0.995)

    wlab = config.wlab
    template = get_s3_object(config.template_bucket, config.template_key, "data")
    template = ants.image_read(template)

    templateL = get_s3_object(config.template_bucket, config.template_label_key, "data")
    templateL = ants.image_read(templateL)

    model_path = get_s3_object(config.model_bucket, config.model_key, "models")
    mdl = tf.keras.models.load_model(model_path)

    havelabels = check_for_labels_in_image( wlab, templateL )

    if not havelabels:
        raise Exception("Label missing from the template")

    output_filename = "outputs/" + config.output_file_prefix
    if not os.path.exists("outputs"):
        os.makedirs('outputs')

    output_filename_sr = output_filename + "_SR.nii.gz"
    output_filename_srOnNativeSeg = output_filename  +  "_srOnNativeSeg.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_nr_seg = output_filename  +  "_NR_seg.nii.gz"

    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
    output_filename_srOnNativeSeg_csv = output_filename  + "_srOnNativeSeg.csv"
    output_filename_nr_seg_csv = output_filename  + "_NR_seg.csv"
    output_filename_ortho_plot_sr = output_filename  +  "_ortho_plot_SR.png"
    output_filename_ortho_plot_srOnNativeSeg = output_filename  +  "_ortho_plot_srOnNativeSeg.png"
    output_filename_ortho_plot_nrseg = output_filename  +  "_ortho_plot_NRseg.png"
    output_filename_ortho_plot_srseg = output_filename  +  "_ortho_plot_SRseg.png"

    output = native_to_superres_ljlf_segmentation(
            target_image = input_image, # n3 image
            segmentation_numbers = wlab,
            template = template,
            template_segmentation = templateL,
            library_intensity = brains,
            library_segmentation = brainsSeg,
            seg_params = config.seg_params,
            seg_params_sr = config.seg_params_sr,
            sr_params = config.sr_params,
            sr_model = mdl,
    )
    # SR Image
    sr = output['srOnNativeSeg']['super_resolution']
    ants.image_write(
            sr,
            output_filename_sr,
    )
    ants.plot_ortho(
            ants.crop_image(sr),
            flat=True,
            filename=output_filename_ortho_plot_sr,
    )

    # SR on native Seg
    srOnNativeSeg = output['srOnNativeSeg']['super_resolution_segmentation']
    ants.image_write(
            srOnNativeSeg,
            output_filename_srOnNativeSeg,
    )
    SRNSdf = ants.label_geometry_measures(srOnNativeSeg)
    SRNSdf.to_csv(output_filename_srOnNativeSeg_csv, index=False)
    cmask = ants.threshold_image(srOnNativeSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(sr, cmask),
            overlay=ants.crop_image(srOnNativeSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_srOnNativeSeg,
    )

    # SR Seg
    srSeg = output['srSeg']['segmentation']
    ants.image_write(
            srSeg,
            output_filename_sr_seg,
    )
    SRdf = ants.label_geometry_measures(srSeg)
    SRdf.to_csv(output_filename_sr_seg_csv, index=False)
    cmask = ants.threshold_image(srSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(sr, cmask),
            overlay=ants.crop_image(srSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_srseg,
    )

    # Native Seg
    nativeSeg = output['nativeSeg']['segmentation']
    ants.image_write(
            nativeSeg,
            output_filename_nr_seg,
    )
    NRdf = ants.label_geometry_measures(nativeSeg)
    NRdf.to_csv(output_filename_nr_seg_csv, index=False)
    cmask = ants.threshold_image(nativeSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(input_image, cmask),
            overlay=ants.crop_image(nativeSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_nrseg,
    )
    if  config.environment == "prod":
        handle_outputs(
            config.input_value,
            config.output_bucket,
            config.output_prefix,
            config.process_name,
        )
    elif config.environment ==  "val":
        ### Set up target image labels ###
        nativeGroundTruth = ants.image_read(target_image_labels_path)
        nativeGroundTruth = ants.mask_image(
            nativeGroundTruth,
            nativeGroundTruth,
            level = wlab,
            binarize=False
        )
        gtSR = super_resolution_segmentation_per_label(
            imgIn = ants.iMath(input_image, "Normalize"),
            segmentation = nativeGroundTruth, # usually, an estimate from a template, not GT
            upFactor = config.sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = config.sr_params['dilation_amount'],
            verbose = config.sr_params['verbose']
        )
        nativeGroundTruthProbSR = gtSR['probability_images'][0]
        nativeGroundTruthSR = gtSR['super_resolution_segmentation']
        nativeGroundTruthBinSR = ants.mask_image(
            nativeGroundTruthSR,
            nativeGroundTruthSR,
            wlab,
            binarize=True
        )
        ######

        srsegLJLF = ants.threshold_image(output['srSeg']['probsum'], 0.5, math.inf )
        nativeOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruth,
            output['nativeSeg']['segmentation']
        )
        srOnNativeOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruthSR,
            output['srOnNativeSeg']['super_resolution_segmentation']
        )
        srOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruthSR,
            output['srSeg']['segmentation']
        )
        srOverlap2 = ants.label_overlap_measures( nativeGroundTruthBinSR, srsegLJLF )

        brainName = []
        dicevalNativeSeg = []
        dicevalSRNativeSeg = []
        dicevalSRSeg = []

        brainName.append(target_image_base)
        dicevalNativeSeg.append(nativeOverlapSloop["MeanOverlap"][0])
        dicevalSRNativeSeg.append( srOnNativeOverlapSloop["MeanOverlap"][0])
        dicevalSRSeg.append( srOverlapSloop["MeanOverlap"][0])

        dict = {
		'name': brainName,
		'diceNativeSeg': dicevalNativeSeg,
		'diceSRNativeSeg': dicevalSRNativeSeg,
		'diceSRSeg': dicevalSRSeg
	}
        df = pd.DataFrame(dict)
        path = f"{target_image_base}_dice_scores.csv"
        df.to_csv("/tmp/" + path, index=False)
        s3 = boto3.client('s3')
        s3.upload_file(
            "/tmp/" + path,
            config.output_bucket,
            config.output_prefix + path,
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

if __name__ == "__main__":
    config = sys.argv[1]
    local_jlf(config)
