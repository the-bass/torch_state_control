import os
import glob
import torch

from .constants import NETWORK_PARAMETERS_SUBDIRECTORY, STATE_CONTROL_DIRECTORY
from .accountant import Accountant


class StateManager:

    def __init__(self, module, name, directory=None, all_onto_cpu=False):
        self.module = module
        self.name = name
        if directory:
            self.storage_directory = directory
        else:
            self.storage_directory = os.path.join(STATE_CONTROL_DIRECTORY, name)
        self.accountant = Accountant(
            directory=self.storage_directory,
            all_onto_cpu=all_onto_cpu
        )
        self.latest_checkpoint = None

    def __getitem__(self, index):
        record = self.accountant[index]
        return self.__load_record__(record)

    def __len__(self):
        return len(self.accountant)

    def __load_checkpoint__(self, record):
        self.latest_checkpoint = record
        self.module.load_state_dict(record.state_dict)

        return record

    def __ensure_directory_exists__(self, directory):
        if os.path.exists(directory):
            return

        os.makedirs(directory)

    def save(self, train_set_performance=None, dev_set_performance=None, losses_since_last_checkpoint=None, notes=None):
        self.__ensure_directory_exists__(self.storage_directory)

        state_dict = self.module.state_dict()
        previous_checkpoint = self.latest_checkpoint.id if self.latest_checkpoint else None

        new_record = self.accountant.new_record(
            notes=notes,
            state_dict=state_dict,
            previous_checkpoint=previous_checkpoint,
            train_set_performance=train_set_performance,
            dev_set_performance=dev_set_performance,
            losses_since_last_checkpoint=losses_since_last_checkpoint
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
