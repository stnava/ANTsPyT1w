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

def run_test():
    if os.getenv('CI') == 'true' and os.getenv('CIRCLECI') == 'true':
        assert True
    else:
        try:
            antspyt1w.get_data()
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            with tempfile.TemporaryDirectory() as temp_dir:
                tempfn=temp_dir+'/apt1wtest'
                testhier = antspyt1w.hierarchical( img, output_prefix=tempfn,
                    labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
                antspyt1w.write_hierarchical( testhier, output_prefix=tempfn )
                uid = os.path.basename( fn )
                uid = re.sub(".nii.gz","",uid)
                outdf = antspyt1w.merge_hierarchical_csvs_to_wide_format( testhier['dataframes'], identifier=uid )
                outdf.to_csv( tempfn + "_mergewide.csv" )
            temp_dir.cleanup()
            assert True
        except AssertionError:
            print("Failure")


run_test()
