class Record:
    def __init__(self, id, id_of_previous_checkpoint, state_dict_storage_id, notes, created_at):
        self.id = id
        self.id_of_previous_checkpoint = id_of_previous_checkpoint
        self.state_dict_storage_id = state_dict_storage_id
        self.notes = notes
        self.created_at = created_at
