import os
import re
import torch


class TorchStoreman:
    """ Capable of persistently storing and fetching Torch tensors in the given
    ´directory´.
    """

    FILE_EXTENSION = 'torch'

    def __init__(self, directory, load_onto_cpu=False):
        self.directory = directory
        self.load_onto_cpu = load_onto_cpu

    def __file_name_for_id__(self, id):
        return f"{id}.{self.FILE_EXTENSION}"

    def __is_torch_file__(self, file_name):
        return not re.search(fr'.\.{self.FILE_EXTENSION}$', file_name) == None

    def __id_from_torch_file_name__(self, file_name):
        file_name_without_extension =  file_name.replace(f'.{self.FILE_EXTENSION}', '')
        id = int(file_name_without_extension)
        return id

    def __highest_existing_id__(self):
        highest_id = -1

        for file_name in os.listdir(self.directory):
            if not self.__is_torch_file__(file_name):
                continue

            state_dict_id = self.__id_from_torch_file_name__(file_name)
            if state_dict_id > highest_id:
                highest_id = state_dict_id

        return highest_id

    def store(self, torch_tensor):
        envisaged_id = self.__highest_existing_id__() + 1
        envisaged_file_name = self.__file_name_for_id__(envisaged_id)
        envisaged_file_path = os.path.join(self.directory, envisaged_file_name)

        torch.save(torch_tensor, envisaged_file_path)

        return envisaged_id

    def fetch(self, id):
        location = os.path.join(self.directory, self.__file_name_for_id__(id))

        load_options = {}
        if self.load_onto_cpu:
            load_options['map_location'] = {'cuda:0': 'cpu'}

        return torch.load(location, **load_options)
