import unittest
import os
import shutil
import torch
import torch_testing as tt
from constants import TEST_TMP_DIRECTORY
from torch_state_control import TorchStoreman


class TestStateDictStoreman(unittest.TestCase):

    def setUp(self):
        self.test_dir = TEST_TMP_DIRECTORY
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(TEST_TMP_DIRECTORY)

    def test___is_torch_file__(self):
        storeman = TorchStoreman(directory=self.test_dir)

        self.assertTrue(storeman.__is_torch_file__('3141.torch'))
        self.assertTrue(storeman.__is_torch_file__('0.torch'))
        self.assertFalse(storeman.__is_torch_file__('3141.torchd'))
        self.assertFalse(storeman.__is_torch_file__('.torch'))
        self.assertFalse(storeman.__is_torch_file__('3141.torc'))

    def test___id_from_torch_file_name__(self):
        storeman = TorchStoreman(directory=self.test_dir)

        self.assertEqual(
            storeman.__id_from_torch_file_name__('3141.torch'),
            3141)

    def test_store_and_fetch(self):
        tensorA = torch.rand(23, 455)
        tensorB = torch.rand(23)
        tensorC = torch.rand(2, 23)

        storeman = TorchStoreman(directory=self.test_dir)

        idA = storeman.store(tensorA)
        idB = storeman.store(tensorB)
        idC = storeman.store(tensorC)

        # Initialize new instance to make sure the data was stored persistently.
        storeman = TorchStoreman(directory=self.test_dir)

        tt.assert_equal(storeman.fetch(idB), tensorB)
        tt.assert_equal(storeman.fetch(idC), tensorC)
        tt.assert_equal(storeman.fetch(idA), tensorA)

if __name__ == '__main__':
    unittest.main()
