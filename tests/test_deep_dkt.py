import antspyt1w
import unittest
import ants
import antspynet

class TestDeepDKT(unittest.TestCase):

    def setUp(self):
        self.input_image = antspynet.get_antsxnet_data( "example_image" )
        template = antspynet.get_antsxnet_data( "biobank" )
        self.template = ants.image_read(template)

    def test_deep_dkt(self):
        input_params={
            "target_image": ants.image_read(self.input_image),
            "template": self.template,
            "output_path": "/tmp/deep_dkt/",
        }
        output = antspyt1w.deep_dkt(**input_params)
        print(output)
        self.assertTrue(output)



if __name__ == "__main__":
    unittest.main()
