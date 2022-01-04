import sys, os
import unittest
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import tempfile
import shutil
import antspyt1w
import antspynet
import ants
import re
import pandas as pd
antspyt1w.get_data(force_download=True)
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
img = ants.image_read( fn )

testingClass = unittest.TestCase( )
with tempfile.TemporaryDirectory() as temp_dir:
    tempfn=temp_dir+'/apt1wtest'
    testhier = antspyt1w.hierarchical( img, output_prefix=tempfn,
        labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
    antspyt1w.write_hierarchical( testhier, output_prefix=tempfn )
    uid = os.path.basename( fn )
    uid = re.sub(".nii.gz","",uid)
    outdf = antspyt1w.merge_hierarchical_csvs_to_wide_format( testhier['dataframes'], identifier=uid )
    outdf.to_csv( tempfn + "_mergewide.csv" )

##### specialized labeling for hippocampus
# hippLR = antspyt1w.deep_hippo( img, templateb, 1 )
# testingClass.assertAlmostEqual(
#    float( hippLR['HLStats']['VolumeInMillimeters'][0]/20000.0 ),
#    float( 2956.0/20000.0 ), 2, "HLStats volume not close enough")

temp_dir.cleanup()

##### specialized labeling for hypothalamus
# FIXME hypothalamus
sys.exit(os.EX_OK) # code 0, all ok
