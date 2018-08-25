import torch.nn as nn
from ..state_manager import StateManager


class StatefulModule(nn.Module):

    def __init__(self, name=None, directory=None, load_onto_cpu=False):
        super().__init__()

        self.state_manager = StateManager(
            module=self,
            name=name,
            directory=directory,
            load_onto_cpu=load_onto_cpu
        )

    def save_checkpoint(self, *args, **kwargs):
        return self.state_manager.save(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        return self.state_manager.load(*args, **kwargs)

    def load_latest_checkpoint(self, *args, **kwargs):
        return self.state_manager.load_latest(*args, **kwargs)

    def latest_checkpoint(self):
        return self.state_manager.latest_checkpoint
