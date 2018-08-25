import os
import csv
import datetime
import math
import torch
import json

from .record import Record
from .constants import ACCOUNT_BOOK


class Accountant:
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    CSV_FIELDS = ['id', 'id_of_previous_checkpoint', 'state_dict_storage_id', 'created_at', 'notes']

    def __init__(self, directory):
        self.directory = directory
        self.account_book = os.path.join(self.directory, ACCOUNT_BOOK)
        self.__refresh_list__()

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)

    def __refresh_list__(self):
        self.list = self.__load_list__()

    def __account_book_exists__(self):
        return os.path.exists(self.account_book)

    def __load_list__(self):
        """ Reads the account book if it exists. """

        dict_list = []

        if not self.__account_book_exists__():
            return dict_list

        with open(self.account_book, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                record = self.__record_from_csv_row__(row)
                dict_list.append(record)

        return dict_list

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

        def load_datetime(csv_value):
            if csv_value == '':
                return None

            return datetime.datetime.strptime(csv_value, self.DATETIME_FORMAT)

        id = load_int(csv_row['id'])
        id_of_previous_checkpoint = load_int(csv_row['id_of_previous_checkpoint'])
        state_dict_storage_id = load_int(csv_row['state_dict_storage_id'])
        notes = json.loads(csv_row['notes'])
        created_at = load_datetime(csv_row['created_at'])

        return Record(
            id=id,
            id_of_previous_checkpoint=id_of_previous_checkpoint,
            state_dict_storage_id=state_dict_storage_id,
            notes=notes,
            created_at=created_at
        )

    def __append_record__(self, record, location):
        parameters = record.__dict__.copy()
        parameters['created_at'] = parameters['created_at'].strftime(self.DATETIME_FORMAT)
        parameters['notes'] = json.dumps(parameters['notes'])

        with open(location, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
            writer.writerow(parameters)

    def __data_file_name__(self, id):
        return f"{id}.pytorch"

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

    def new_record(self, state_dict_storage_id, id_of_previous_checkpoint=None, notes=None):
        # index
        indices = list(map(lambda x: x.id, self.list))
        highest_idx = max(indices) if len(indices) > 0 else -1
        idx = highest_idx + 1

        # created_at
        created_at = datetime.datetime.utcnow()

        new_record = Record(
            id=idx,
            id_of_previous_checkpoint=id_of_previous_checkpoint,
            state_dict_storage_id=state_dict_storage_id,
            notes=notes,
            created_at=created_at
        )

        self.__ensure_account_book_exists__()
        self.__append_record__(new_record, location=self.account_book)

        self.__refresh_list__()

        return new_record

    def record_by_id(self, id):
        for record in self.list:
            if record.id == id:
                return record

        return None
