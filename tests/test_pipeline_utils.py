import boto3
from moto import mock_s3
from superiq import pipeline_utils
import unittest
import json
import os

class TestPipelineUtils(unittest.TestCase):
    
    def setUp(self):
        real_path = os.path.realpath("..")
        self.test_config_path = os.path.join(real_path, "superiq/configs/test_config.json")
        with open(self.test_config_path, "r") as c: 
            self.test_config =  str(json.load(c))
    
    @mock_s3
    def test_get_s3_object(self):
        conn = boto3.client('s3')
        conn.create_bucket(Bucket="bucket")
        key = "/test/prefix/test_object.txt"
        conn.put_object(Bucket="bucket", Key=key, Body="test")
        local_dir = "/tmp"
        local_path = pipeline_utils.get_s3_object("bucket", key, local_dir)   
        expected_filepath = "test_object.txt" 
        files = os.listdir('/tmp')
        self.assertIn(expected_filepath, files)
        os.remove(local_path) 

    def test_LoadConfig(self):
        config = pipeline_utils.LoadConfig(self.test_config)
        self.assertTrue(config.input_value) 

    @mock_s3
    def test_handle_outputs(self):
        conn = boto3.client('s3')
        input_object = "test/input/object/key.nii.gz" 
        output_bucket = "bucket"
        output_prefix = "test/prefix/"
        conn.create_bucket(Bucket=output_bucket)
        pipeline_utils.handle_outputs(
                input_object,
                output_bucket,
                output_prefix,
                "test_process",
                local_output_dir="configs",
        )

    @mock_s3
    def test_get_pipeline_data(self):
        conn = boto3.client('s3')
        conn.create_bucket(Bucket="bucket")
        key = "base/test/prefix/test_object.nii.gz"
        conn.put_object(Bucket="bucket", Key=key, Body="test")
        
        local_dir = "/tmp"
        filename = "object.nii.gz" 
        initial_object = "base/test-prefix.nii.gz" 
        local_path = pipeline_utils.get_pipeline_data(
                filename,
                initial_object,
                "bucket",
                "base/"
        )
        expected_local_path = "data/test_object.nii.gz"
        self.assertEqual(expected_local_path, local_path)
        files = os.listdir('/tmp')
        os.remove(local_path) 

    def test_derive_s3_path(self):
        input_s3_path = "some/test/key/file-name-here.nii.gz"
        path, basename = pipeline_utils.derive_s3_path(input_s3_path)
        self.assertEqual(path, "file/name/here/")
        self.assertEqual(basename, "file-name-here")

    @mock_s3
    def test_list_images(self):
        conn = boto3.client('s3')
        conn.create_bucket(Bucket="bucket")
        key = "base/test/prefix/test_object.nii.gz"
        conn.put_object(Bucket="bucket", Key=key, Body="test")
   
        bucket = "bucket"
        prefix = "base/" 
        keys = pipeline_utils.list_images(bucket, prefix)
        self.assertIn(key, keys)


if __name__ == "__main__":
    unittest.main()
