import torch.nn as nn
from ..state_manager import StateManager
from ..tracer import Tracer


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

    def checkpoint_backtrace(self, checkpoint_id=None):
        if checkpoint_id == None:
            latest_checkpoint = self.latest_checkpoint()
            checkpoint_id = latest_checkpoint.id if latest_checkpoint else None

        if checkpoint_id == None:
            return []

        tracer = Tracer(directory=self.state_manager.storage_directory)

        return tracer.backtrace_for(checkpoint_id)

    def checkpoint(self, checkpoint_id=None):
        if checkpoint_id == None:
            return self.latest_checkpoint()

        tracer = Tracer(directory=self.state_manager.storage_directory)

        if checkpoint_id < 0:
            return tracer.accountant[checkpoint_id]

        return tracer.accountant.record_by_id(checkpoint_id)
