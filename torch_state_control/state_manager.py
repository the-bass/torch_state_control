import os
import glob
import torch

from .constants import STATE_DICTS_SUBDIRECTORY, STATE_CONTROL_DIRECTORY
from .accountant import Accountant
from .torch_storeman import TorchStoreman


class StateManager:
    """ Manages different values of a PyTorch module's parameters. """

    def __init__(self, module, name=None, directory=None, load_onto_cpu=False):
        self.module = module

        # Default the ´name´ to the name of the class the given module is an
        # instance of.
        self.name = name if name else type(module).__name__

        # Use the default directory if no directory is given.
        self.storage_directory = directory if directory else os.path.join(STATE_CONTROL_DIRECTORY, self.name)
        self.state_dicts_directory = os.path.join(self.storage_directory, STATE_DICTS_SUBDIRECTORY)

        self.accountant = Accountant(directory=self.storage_directory)
        self.state_dict_storeman = TorchStoreman(
            directory=self.state_dicts_directory,
            load_onto_cpu=load_onto_cpu)
        self.latest_checkpoint = None

    def __getitem__(self, index):
        record = self.accountant[index]
        return self.__load_record__(record)

    def __len__(self):
        return len(self.accountant)

    def __load_checkpoint__(self, record):
        state_dict = self.state_dict_storeman.fetch(record.state_dict_storage_id)
        self.module.load_state_dict(state_dict)
        self.latest_checkpoint = record

        return record

    @staticmethod
    def __ensure_directory_exists__(directory):
        if os.path.exists(directory):
            return

        os.makedirs(directory)

    def __ensure_directories_exist__(self):
        self.__ensure_directory_exists__(self.storage_directory)
        self.__ensure_directory_exists__(self.state_dicts_directory)

    def save(self, notes=None):
        self.__ensure_directories_exist__()

        id_of_previous_checkpoint = self.latest_checkpoint.id if self.latest_checkpoint else None
        current_state_dict = self.module.state_dict()
        state_dict_storage_id = self.state_dict_storeman.store(current_state_dict)

        new_record = self.accountant.new_record(
            id_of_previous_checkpoint=id_of_previous_checkpoint,
            state_dict_storage_id=state_dict_storage_id,
            notes=notes
        )

        self.latest_checkpoint = new_record

        return new_record

    def load_latest(self):
        latest_record = self.accountant.latest()

        if not latest_record:
            return

        return self.__load_checkpoint__(latest_record)

    def load(self, id):
        record = self.accountant.record_by_id(id)

        if not record:
            raise Exception(f"Record with id {id} does not exist.")

        return self.__load_checkpoint__(record)
