import os
import csv
import datetime
import math
import torch

from .record import Record
from .constants import ACCOUNT_BOOK, NETWORK_PARAMETERS_SUBDIRECTORY, LOSSES_SUBDIRECTORY


class Accountant:

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    CSV_FIELDS = ['id', 'previous_checkpoint', 'train_set_performance', 'dev_set_performance', 'created_at', 'notes']

    def __init__(self, directory, all_onto_cpu):
        self.directory = directory
        self.all_onto_cpu = all_onto_cpu
        self.account_book = os.path.join(self.directory, ACCOUNT_BOOK)
        self.__refresh_list__()

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)

    def __load_list__(self):
        dict_list = []

        if not self.__account_book_exists__():
            return dict_list

        with open(self.account_book, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                record = self.__record_from_csv_row__(row)
                dict_list.append(record)

        return dict_list

    def __refresh_list__(self):
        self.list = self.__load_list__()

    def __account_book_exists__(self):
        return os.path.exists(self.account_book)

    def __ensure_account_book_exists__(self):
        if self.__account_book_exists__():
            return

        self.__write_headers__(location=self.account_book)

    def __write_headers__(self, location):
        with open(location, 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

    def __record_from_csv_row__(self, csv_row):
        def load_int(csv_value):
            if csv_value == '':
                return None

            return int(csv_value)

        def load_float(csv_value):
            if csv_value == '':
                return None

            return float(csv_value)

        def load_datetime(csv_value):
            if csv_value == '':
                return None

            return datetime.datetime.strptime(csv_value, self.DATETIME_FORMAT)

        id = load_int(csv_row['id'])
        notes = csv_row['notes']
        previous_checkpoint = load_int(csv_row['previous_checkpoint'])
        train_set_performance = load_float(csv_row['train_set_performance'])
        dev_set_performance = load_float(csv_row['dev_set_performance'])
        created_at = load_datetime(csv_row['created_at'])
        state_dict = self.__fetch_data__(NETWORK_PARAMETERS_SUBDIRECTORY, id)
        losses_since_last_checkpoint = self.__fetch_data__(LOSSES_SUBDIRECTORY, id)

        return Record(
            id=id,
            previous_checkpoint=previous_checkpoint,
            state_dict=state_dict,
            train_set_performance=train_set_performance,
            dev_set_performance=dev_set_performance,
            losses_since_last_checkpoint=losses_since_last_checkpoint,
            notes=notes,
            created_at=created_at
        )

    def __append_record__(self, record, location):
        parameters = record.__dict__.copy()
        parameters['created_at'] = parameters['created_at'].strftime(self.DATETIME_FORMAT)

        parameters.pop('state_dict')
        parameters.pop('losses_since_last_checkpoint')

        if not parameters['train_set_performance'] or math.isnan(parameters['train_set_performance']):
            parameters['train_set_performance'] = ''
        if not parameters['dev_set_performance'] or math.isnan(parameters['dev_set_performance']):
            parameters['dev_set_performance'] = ''

        with open(location, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
            writer.writerow(parameters)

    def __ensure_directory_exists__(self, directory):
        if os.path.exists(directory):
            return

        os.makedirs(directory)

    def __data_file_name__(self, id):
        return f"{id}.pytorch"

    def __store_data__(self, data, subdirectory, id):
        dir = os.path.join(self.directory, subdirectory)
        self.__ensure_directory_exists__(dir)
        location = os.path.join(dir, self.__data_file_name__(id))
        torch.save(data, location)

    def __fetch_data__(self, subdirectory, id):
        location = os.path.join(self.directory, subdirectory, self.__data_file_name__(id))

        load_options = {}

        if self.all_onto_cpu:
            load_options['map_location'] = lambda storage, loc: storage

        return torch.load(location, **load_options)

    def any_exist(self):
        return len(self.list) > 0

    def latest(self):
        if not self.any_exist():
            return None

        return self.list[-1]

    def oldest(self):
        if not self.any_exist():
            return None

        return self.list[0]

    def new_record(self, state_dict, notes=None, previous_checkpoint=None, train_set_performance=None, dev_set_performance=None, losses_since_last_checkpoint=None):
        # index
        indices = list(map(lambda x: x.id, self.list))
        highest_idx = max(indices) if len(indices) > 0 else -1
        idx = highest_idx + 1

        # created_at
        created_at = datetime.datetime.utcnow()

        new_record = Record(
            id=idx,
            previous_checkpoint=previous_checkpoint,
            state_dict=state_dict,
            train_set_performance=train_set_performance,
            dev_set_performance=dev_set_performance,
            losses_since_last_checkpoint=losses_since_last_checkpoint,
            notes=notes,
            created_at=created_at
        )

        self.__store_data__(state_dict, NETWORK_PARAMETERS_SUBDIRECTORY, new_record.id)
        self.__store_data__(losses_since_last_checkpoint, LOSSES_SUBDIRECTORY, new_record.id)

        self.__ensure_account_book_exists__()
        self.__append_record__(new_record, location=self.account_book)

        self.__refresh_list__()

        return new_record

    # def remove(self, record_id):
    #     new_account_book_location = self.account_book + 'NEW'
    #     self.__write_headers__(location=new_account_book_location)
    #
    #     for record in self.list:
    #         if record.id == record_id:
    #             continue
    #
    #         self.__append_record__(record, location=new_account_book_location)
    #
    #     os.remove(self.account_book)
    #     os.rename(new_account_book_location, self.account_book)
    #
    #     self.__refresh_list__()

    def record_by_id(self, id):
        for record in self.list:
            if record.id == id:
                return record

        return None
