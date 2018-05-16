import unittest
import os
import shutil

from torch_state_control.analysis.analyst import Analyst
from fixtures.super_simple_stateful_net import SuperSimpleStatefulNet
from constants import TEST_TMP_DIRECTORY


class TestAnalyst(unittest.TestCase):

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

        # Create some checkpoints.
        net.save_checkpoint(
            train_set_performance=1,
            dev_set_performance=2,
            losses_since_last_checkpoint=[0.9, 0.8, 0.7, 0.6],
            notes='Learning rate: 0.01'
        )
        net.save_checkpoint(
            train_set_performance=2,
            dev_set_performance=3,
            losses_since_last_checkpoint=[0.5, 0.4],
            notes='Learning rate: 0.005'
        )
        net.save_checkpoint(
            train_set_performance=3,
            dev_set_performance=4,
            losses_since_last_checkpoint=[0.3, 0.2, 0.1],
            notes='Learning rate: 0.001'
        )

        analyst = Analyst(name='simple_stateful_net', directory=self.test_dir)

        analyst.plot_performances(2)

if __name__ == '__main__':
    unittest.main()
