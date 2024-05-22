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
# just test that things loaded ok
if os.getenv('CI') == 'true' and os.getenv('CIRCLECI') == 'true':
    def test_simple():
        assert os.getenv('CI') == 'true' and os.getenv('CIRCLECI') == 'true'
    def test_download():
        assert antspyt1w.get_data() == None
    def test_img():
        fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
        img = ants.image_read( fn )
        assert img is not None
    def test_run():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            testhier = antspyt1w.hierarchical( img, output_prefix=tempfn,
                        labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
            assert testhier is not None
else:
    def test_simple():
        assert os.getenv('CI') != 'true' and os.getenv('CIRCLECI') != 'true'
