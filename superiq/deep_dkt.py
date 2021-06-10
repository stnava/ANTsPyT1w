import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string
from superiq.pipeline_utils import *

def deep_dkt(
    target_image,
    segmentation_numbers,
    template,
    sr_model,
    sr_params,
    output_path=None,
):
    """

    """
    img = target_image

    rig = ants.registration( template, img, "Rigid" )
    rigi = rig['warpedmovout']

    mdl = tf.keras.models.load_model(sr_model)

    dkt = antspynet.desikan_killiany_tourville_labeling(
        rigi,
        do_preprocessing=False,
        return_probability_images=True,
    )

    segorigspace = ants.apply_transforms(
        img,
        dkt['segmentation_image'],
        rig['fwdtransforms'],
        whichtoinvert=[True],
        interpolator='genericLabel'
    )

    srseg = super_resolution_segmentation_per_label(
        imgIn = img,
        segmentation = segorigspace,
        upFactor = [2,2,2],
        sr_model = mdl,
        segmentation_numbers = segmentation_numbers,
        dilation_amount = 6,
        verbose = True
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = output_path + "deep_dkt_Labels" + list_to_string(segmentation_numbers)
    print(output_filename)
    output_filename_native = output_filename + "_OR_seg.nii.gz"
    output_filename_native_csv = output_filename + "_OR_seg.csv"
    ants.image_write( segorigspace, output_filename_native )
    label_data = ants.label_geometry_measures(segorigspace, img)
    label_data = label_data.loc[label_data['Label'].isin(segmentation_numbers)]
    label_data.to_csv(output_filename_native_csv, index=False)

    output_filename_sr = output_filename + "_SR.nii.gz"
    ants.image_write( srseg['super_resolution'], output_filename_sr )
    output_filename_sr_seg = output_filename +  "_SR_seg.nii.gz"
    ants.image_write(srseg['super_resolution_segmentation'], output_filename_sr_seg )
    output_filename_sr_seg_csv = output_filename + "_SR_seg.csv"
    label_data_sr = ants.label_geometry_measures(
        srseg['super_resolution_segmentation'],
        srseg['super_resolution'],
    )
    label_data_sr.to_csv(output_filename_sr_seg_csv, index=False)

    return  {
        "nativeSeg": ants.image_read(output_filename_native), #segorigspace
        "superresSeg": ants.image_read(output_filename_sr_seg),
        "superres": ants.image_read(output_filename_sr),
        "labels_or": output_filename_native_csv,
        "labels_sr": output_filename_sr_seg_csv,
    }

