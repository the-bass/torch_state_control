import unittest
import numpy.testing
import os
import shutil
import torch
import datetime

from torch_state_control import StateManager
from torch_state_control.analysis.tracer import Tracer
from fixtures.super_simple_net import SuperSimpleNet
import constants


class TestTracer(unittest.TestCase):

    def setUp(self):
        self.test_dir = constants.TEST_TMP_DIRECTORY
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(constants.TEST_TMP_DIRECTORY)

    def test_history(self):
        # Initialize.
        net = SuperSimpleNet()
        manager = StateManager(net=net, name='simple_net', directory=self.test_dir)

        # Create a few checkpoints.
        manager.save()
        manager.save()
        manager.save()

        # Reset.
        net = SuperSimpleNet()
        manager = StateManager(net=net, name='simple_net', directory=self.test_dir)

        # Load second checkpoint.
        manager.load(1)

        # Create one more checkpoint.
        manager.save()

        # Check history.
        tracer = Tracer(directory=self.test_dir)
        backtrace = tracer.backtrace_for(3)

        self.assertEqual(len(backtrace), 3)
        self.assertEqual(backtrace[0].id, 0)
        self.assertEqual(backtrace[1].id, 1)
        self.assertEqual(backtrace[2].id, 3)
