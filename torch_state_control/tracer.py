import os

from .accountant import Accountant


class Tracer:
    def __init__(self, directory):
        self.directory = directory
        self.accountant = Accountant(directory=self.directory)

    def __prepend_previous_records__(self, array):
        id_of_previous_checkpoint = array[0].id_of_previous_checkpoint

        if id_of_previous_checkpoint == None:
            return array

        previous_record = self.accountant.record_by_id(id_of_previous_checkpoint)
        array.insert(0, previous_record)

        return self.__prepend_previous_records__(array)

    def backtrace_for(self, checkpoint_id):
        if checkpoint_id >= 0:
            last_record = self.accountant.record_by_id(checkpoint_id)
        else:
            last_record = self.accountant[checkpoint_id]

        records = []

        if not last_record:
            return []

        records = [last_record]

        return self.__prepend_previous_records__(records)

    def latest(self):
        return self.accountant.latest()
