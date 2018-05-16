import os

from .tracer import Tracer
from ..constants import STATE_CONTROL_DIRECTORY


class Analyst:

    def __init__(self, name, directory=None):
        if directory:
            self.directory = directory
        else:
            self.directory = os.path.join(STATE_CONTROL_DIRECTORY, name)

    def __records__(self, checkpoint):
        tracer = Tracer(directory=self.directory)
        checkpoint_id = checkpoint

        if not checkpoint_id:
            checkpoint_id = tracer.latest().id

        return tracer.backtrace_for(checkpoint_id)

    def changelog(self, checkpoint=None):
        records = self.__records__(checkpoint)

        unique_notes = []

        for record in records:
            if len(unique_notes) > 0 and unique_notes[-1]['notes'] == record.notes:
                continue

            unique_notes.append({
                'checkpoint_id': record.id,
                'notes': record.notes
            })

        return unique_notes

    def performances(self, checkpoint=None):
        records = self.__records__(checkpoint)

        train_set_performances = []
        dev_set_performances = []
        for record in records:
            train_set_performances.append(record.train_set_performance)
            dev_set_performances.append(record.dev_set_performance)

        return train_set_performances, dev_set_performances

    def losses(self, checkpoint=None):
        records = self.__records__(checkpoint)

        losses = []
        for record in records:
            losses += record.losses_since_last_checkpoint

        return losses
