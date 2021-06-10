import superiq
from superiq.pipeline_utils import *
import unittest
import ants
import antspynet

class TestDeepDKT(unittest.TestCase):

    def setUp(self):
        self.original_image = get_s3_object(
            "invicro-data-shared",
            "ADNI/127_S_0112/20160205/T1w/001/ADNI-127_S_0112-20160205-T1w-001.nii.gz",
            "/tmp",
        )
        self.input_image = get_pipeline_data(
            'n3.nii.gz',
            self.original_image,
            "eisai-basalforebrainsuperres2",
            "superres-pipeline/",
            "/tmp"
        )
        template = antspynet.get_antsxnet_data("biobank")
        self.template = ants.image_read(template)

        self.sr_model = get_s3_object(
            "invicro-pipeline-inputs",
            "models/SEGSR_32_ANINN222_3.h5",
            "/tmp"
        )

        self.seg_numbers = [1006, 1007, 1015, 1016,]
        self.sr_params = {
            "upFactor": [2,2,2,],
            "dilation_amount": 6,
            "verbose": True,
        }

    def test_deep_dkt(self):
        input_params={
            "target_image": ants.image_read(self.input_image),
            "segmentation_numbers": self.seg_numbers,
            "template": self.template,
            "sr_model": self.sr_model,
            "sr_params": self.sr_params,
            "output_path": "/tmp/deep_dkt/",
        }
        output = superiq.deep_dkt(**input_params)
        print(output)
        handle_outputs(
            self.original_image,
            "eisai-basalforebrainsuperres2",
            "superres-pipeline-dkt/",
            'deep_dkt',
            output
        )
        self.assertTrue(output)



if __name__ == "__main__":
    unittest.main()
