class Record:
    def __init__(self, id, state_dict, notes, previous_checkpoint, train_set_performance, dev_set_performance, losses_since_last_checkpoint, created_at):
        self.id = id
        self.previous_checkpoint = previous_checkpoint
        self.state_dict = state_dict
        self.train_set_performance = train_set_performance
        self.dev_set_performance = dev_set_performance
        self.losses_since_last_checkpoint = losses_since_last_checkpoint
        self.created_at = created_at
        self.notes = notes
