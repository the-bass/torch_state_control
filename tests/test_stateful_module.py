import unittest
import numpy.testing
import os
import shutil
import torch

from fixtures.super_simple_stateful_net import SuperSimpleStatefulNet
from constants import TEST_TMP_DIRECTORY


class TestStatefulModule(unittest.TestCase):

    def setUp(self):
        self.test_dir = TEST_TMP_DIRECTORY
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(TEST_TMP_DIRECTORY)

    def test_save_and_load(self):
        # Initialize.
        net = SuperSimpleStatefulNet(
            name='simple_stateful_net',
            directory=self.test_dir
        )

        # Update the weight.
        new_weight = torch.Tensor([[1, 1]])
        net.state_dict()['fc.weight'].copy_(new_weight)

        # Create a checkpoint.
        net.save_checkpoint()

        # Reset net.
        net = SuperSimpleStatefulNet(
            name='simple_stateful_net',
            directory=self.test_dir
        )

        # Before loading a checkpoint, the weights should be reset.
        current_weight = net.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[0, 0]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Load the latest checkpoint.
        net.load_latest_checkpoint()

        # The weights should now be as saved.
        current_weight = net.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[1, 1]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Update the weight once more.
        new_weight = torch.Tensor([[2, 2]])
        net.state_dict()['fc.weight'].copy_(new_weight)

        # Create another checkpoint.
        last_checkpoint = net.save_checkpoint()

        # Update the weight once more.
        new_weight = torch.Tensor([[3, 3]])
        net.state_dict()['fc.weight'].copy_(new_weight)

        # Check the weights.
        current_weight = net.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[3, 3]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Load last saved checkpoint.
        net.load_checkpoint(last_checkpoint.id)

        # Check the weights.
        current_weight = net.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[2, 2]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

    def test_latest_checkpoint(self):
        # Initialize.
        net = SuperSimpleStatefulNet(
            name='simple_stateful_net',
            directory=self.test_dir
        )

        # Create a checkpoint.
        net.save_checkpoint()
        net.save_checkpoint()
        net.save_checkpoint()
        latest_checkpoint = net.save_checkpoint()

        net.latest_checkpoint()
        self.assertEqual(net.latest_checkpoint(), latest_checkpoint)

if __name__ == '__main__':
    unittest.main()
