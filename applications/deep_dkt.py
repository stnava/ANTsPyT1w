import ants
import antspynet
import tensorflow as tf
import numpy as np

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string

template = antspynet.get_antsxnet_data( "biobank" )
template = ants.image_read( template )
template = template * antspynet.brain_extraction( template )

img = ants.image_read( "/Users/stnava/code/super_resolution_pipelines/data/ADNI-127_S_0112-20160205-T1w-001-brain_ext-bxtreg_n3.nii.gz" )

if not 'rig' in locals():
    rig = ants.registration( template, img, "Affine" )
    rigi = rig['warpedmovout']

valid_labels = (
  "4:left lateral ventricle", "5:left inferior lateral ventricle",
  "6:left cerebellem exterior", "7:left cerebellum white matter",
  "10:left thalamus proper", "11:left caudate",
  "12:left putamen", "13:left pallidium",
  "15:4th ventricle", "16:brain stem",
  "17:left hippocampus", "18:left amygdala",
  "24:CSF", "25:left lesion", "26:left accumbens area", "28:left ventral DC",
  "30:left vessel",
  "43:right lateral ventricle", "44:right inferior lateral ventricle",
  "45:right cerebellum exterior", "46:right cerebellum white matter",
  "49:right thalamus proper", "50:right caudate",
  "51:right putamen", "52:right palladium",
  "53:right hippocampus",
  "54:right amygdala",
  "57:right lesion",
  "58:right accumbens area",
  "60:right ventral DC",
  "62:right vessel",
  "72:5th ventricle", "85:optic chasm",
  "91:left basal forebrain",
  "92:right basal forebrain",
  "630:cerebellar vermal lobules I-V",
  "631:cerebellar vermal lobules VI-VII",
  "632:cerebellar vermal lobules VIII-X",
  "1002:left caudal anterior cingulate",
  "1003:left caudal middle frontal",
  "1005:left cuneus",
  "1006:left entorhinal",
  "1007:left fusiform",
  "1008:left inferior parietal",
  "1009:left inferior temporal",
  "1010:left isthmus cingulate",
  "1011:left lateral occipital",
  "1012:left lateral orbitofrontal",
  "1013:left lingual",
  "1014:left medial orbitofrontal",
  "1015:left middle temporal",
  "1016:left parahippocampal",
  "1017:left paracentral",
  "1018:left pars opercularis",
  "1019:left pars orbitalis",
  "1020:left pars triangularis",
  "1021:left pericalcarine",
  "1022:left postcentral",
  "1023:left posterior cingulate",
  "1024:left precentral",
  "1025:left precuneus",
  "1026:left rostral anterior cingulate",
  "1027:left rostral middle frontal",
  "1028:left superior frontal",
  "1029:left superior parietal",
  "1030:left superior temporal",
  "1031:left supramarginal",
  "1034:left transverse temporal",
  "1035:left insula",
  "2002:right caudal anterior cingulate",
  "2003:right caudal middle frontal",
  "2005:right cuneus",
  "2006:right entorhinal",
  "2007:right fusiform",
  "2008:right inferior parietal",
  "2009:right inferior temporal",
  "2010:right isthmus cingulate",
  "2011:right lateral occipital",
  "2012:right lateral orbitofrontal",
  "2013:right lingual",
  "2014:right medial orbitofrontal",
  "2015:right middle temporal",
  "2016:right parahippocampal",
  "2017:right paracentral",
  "2018:right pars opercularis",
  "2019:right pars orbitalis",
  "2020:right pars triangularis",
  "2021:right pericalcarine",
  "2022:right postcentral",
  "2023:right posterior cingulate",
  "2024:right precentral",
  "2025:right precuneus",
  "2026:right rostral anterior cingulate",
  "2027:right rostral middle frontal",
  "2028:right superior frontal",
  "2029:right superior parietal",
  "2030:right superior temporal",
  "2031:right supramarginal",
  "2034:right transverse temporal",
  "2035:right insula" )

# see help for meaning of labels
if not 'dkt' in locals():
    dkt = antspynet.desikan_killiany_tourville_labeling( rigi,
      do_preprocessing=False,
      return_probability_images=True ) # FIXME - use probability images later


if not 'segorigspace' in locals():
    segorigspace = ants.apply_transforms( img, dkt['segmentation_image'],
      rig['fwdtransforms'], whichtoinvert=[True], interpolator='genericLabel')


# OUTPUT: write the native resolution image => segorigspace
# and its label geometry csv
output_filename = "/tmp/deep_dkt/deep_dkt_" # + list_to_string(mysegnumbers)
output_filename_native = output_filename + "_OR_seg.nii.gz"
output_filename_native_csv = output_filename + "_OR_seg.csv"
ants.image_write( segorigspace, output_filename_native )

# NOTE: the code below is SR specific and should only be run in that is requested
########################################
mysegnumbers = [ 1006, 1007, 1015, 1016] # Eisai cortical regions, left
mysegnumbers = [ 2006, 2007, 2015, 2016] # Eisai cortical regions, right
# FIXME - check that mysegnumbers are in valid_labels
########################################
# find the right probability image
mdl = tf.keras.models.load_model( "models/SEGSR_32_ANINN222_3.h5" ) # FIXME - parameterize this

srseg = super_resolution_segmentation_per_label(
    imgIn = img,
    segmentation = segorigspace,
    upFactor = [2,2,2],
    sr_model = mdl,
    segmentation_numbers = mysegnumbers,
    dilation_amount = 6,
    verbose = True
)

# writing ....
output_filename_sr = output_filename + list_to_string(mysegnumbers) + "_SR.nii.gz"
ants.image_write( srseg['super_resolution'], output_filename_sr )
output_filename_sr_seg = output_filename +list_to_string(mysegnumbers) +  "_SR_seg.nii.gz"
ants.image_write(srseg['super_resolution_segmentation'], output_filename_sr_seg )
output_filename_sr_seg_csv = output_filename + list_to_string(mysegnumbers) + "_SR_seg.csv"
# FIXME: write csv here
