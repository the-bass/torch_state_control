import unittest
import numpy.testing
import os
import shutil
import torch
import datetime

from torch_state_control import StateManager
from fixtures.super_simple_net import SuperSimpleNet
from constants import TEST_TMP_DIRECTORY


class TestStateManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = TEST_TMP_DIRECTORY
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(TEST_TMP_DIRECTORY)

    def test_save_load_and_load_latest(self):
        # Initialize.
        module = SuperSimpleNet()
        manager = StateManager(module=module, directory=self.test_dir)

        # Update the weight.
        new_weight = torch.Tensor([[1, 1]])
        module.state_dict()['fc.weight'].copy_(new_weight)

        # Create a checkpoint.
        manager.save()

        # Reset module.
        module = SuperSimpleNet()
        manager = StateManager(module=module, directory=self.test_dir)

        # Before loading a checkpoint, the weights should be reset.
        current_weight = module.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[0, 0]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Load the latest checkpoint.
        manager.load_latest()

        # The weights should now be as saved.
        current_weight = module.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[1, 1]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Update the weight once more.
        new_weight = torch.Tensor([[2, 2]])
        module.state_dict()['fc.weight'].copy_(new_weight)

        # Create another checkpoint.
        latest_checkpoint = manager.save()

        # Update the weight once more.
        new_weight = torch.Tensor([[3, 3]])
        module.state_dict()['fc.weight'].copy_(new_weight)

        # Check the weights.
        current_weight = module.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[3, 3]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

        # Load the latest checkpoint.
        manager.load(latest_checkpoint.id)

        # Check the weights.
        current_weight = module.state_dict()['fc.weight']
        expected_weight = torch.Tensor([[2, 2]])
        numpy.testing.assert_array_equal(current_weight, expected_weight)

    def test_exception_raised_when_load_id_not_exists(self):
        # Initialize.
        module = SuperSimpleNet()
        manager = StateManager(module=module, directory=self.test_dir)

        with self.assertRaisesRegex(Exception, 'Record .* not exist') as cm:
            manager.load(1)

    def test_saving_meta_data(self):
        # Initialize.
        module = SuperSimpleNet()
        manager = StateManager(module=module, directory=self.test_dir)

        # Update the weight.
        new_weight = torch.Tensor([[1, 1]])
        module.state_dict()['fc.weight'].copy_(new_weight)

        meta_data = {
            'train_set_performance': '1.23',
            'dev_set_performance': '1|2|3',
            'losses_since_last_checkpoint': [1.23, 2.34, 3.45],
            'notes': 'Learning rate: 1.23'
        }

        # Create a checkpoint.
        manager.save(**meta_data)

        # Reload.
        module = SuperSimpleNet()
        manager = StateManager(module=module, name='simple_net', directory=self.test_dir)

        # Load the latest checkpoint.
        checkpoint = manager.load_latest()

        # Check that the additional data can be accessed.
        self.assertEqual(checkpoint.id, 0)
        assert(isinstance(checkpoint.created_at, datetime.datetime))
        self.assertLess((datetime.datetime.utcnow() - checkpoint.created_at).seconds, 1)
        self.assertEqual(
            checkpoint.train_set_performance,
            meta_data['train_set_performance']
        )
        self.assertEqual(
            checkpoint.dev_set_performance,
            meta_data['dev_set_performance']
        )
        self.assertEqual(
            checkpoint.notes,
            meta_data['notes']
        )
        self.assertEqual(
            checkpoint.losses_since_last_checkpoint,
            meta_data['losses_since_last_checkpoint']
        )

if __name__ == '__main__':
    unittest.main()
