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
else:
    def test_simple():
        assert os.getenv('CI') != 'true' and os.getenv('CIRCLECI') != 'true'
