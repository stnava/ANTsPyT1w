import os
# set number of threads - this should be optimized for your compute instance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import pandas as pd
from superiq.pipeline_utils import *
import superiq
import unittest

class TestSegmentationValidation(unittest.TestCase):
    def create_test_params(self):
        tdir = "/tmp/segmentation_validation"
        if ( not path. exists( tdir ) ):
            os.makdirs(tdir)
            #raise RuntimeError('Failed to find the data directory')

        print("====> Getting remote data")

        template_bucket = "invicro-pipeline-inputs"
        template_key = "adni_templates/adni_template.nii.gz"
        template_label_key = "adni_templates/adni_template_dkt_labels.nii.gz"
        model_bucket = "invicro-pipeline-inputs"
        model_key = "models/SEGSR_32_ANINN222_3.h5"
        atlas_bucket = "invicro-pipeline-inputs"
        atlas_image_prefix = "OASIS30/Brains/"
        atlas_label_prefix = "OASIS30/SegmentationsJLFOR/"

        template = get_s3_object(template_bucket, template_key, tdir)
        templateL = get_s3_object(template_bucket, template_label_key, tdir)
        model_path = get_s3_object(model_bucket, model_key, tdir)

        atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
        brains = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_image_keys]
        brains.sort()
        #brains = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
        brainsSeg = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_label_keys]
        brainsSeg.sort()

        seg_params={
            'submask_dilation': 8,
            'reg_iterations': [100, 100, 20],
            'searcher': 1,
            'radder': 2,
            'syn_sampling': 32,
            'syn_metric': 'mattes',
            'max_lab_plus_one': True, 'verbose': False}

        seg_params_sr={
            'submask_dilation': seg_params['submask_dilation']*1,
            'reg_iterations': seg_params['reg_iterations'],
            'searcher': seg_params['searcher'],
            'radder': seg_params['radder'],
            'syn_sampling': seg_params['syn_sampling'],
            'syn_metric': seg_params['syn_metric'],
            'max_lab_plus_one': True, 'verbose': False}

        sr_params={
            "upFactor": [2,2,2],
            "dilation_amount": seg_params["submask_dilation"],
            "verbose":True
        }

        wlab = [75,76] # basal forebraina

        # An example parameters argument
        test_params = {
            "target_image": "",
            "segmentation_numbers": wlab,
            "template": template,
            "template_segmentation": templateL,
            "library_intensity": "",
            "library_segmentation": "",
            "seg_params": seg_params,
            "seg_params_sr": seg_params_sr,
            "sr_params": sr_params,
            "sr_model": model_path,
            "forward_transforms": None,
        }
        return test_params

    def test_leave_one_out_cross_validation(self):
        pass

    def test_make_validation_pools(self):
        test_params = {'a': 1, 'b': 2}
        test_list_a = [2,3,5,7]
        test_list_b = ['a','b','c','d']
        def square_it(x):
            return x*x
        test_pools = superiq.make_validation_pools(
            square_it,
            test_params,
            test_list_a,
            test_list_b,
        )
        self.assertEqual(len(test_pools),len(test_list_a))


if __name__ == "__main__":
    unittest.main()
