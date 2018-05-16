import torch.nn as nn
from torch_state_control.nn import StatefulModule


class SuperSimpleStatefulNet(StatefulModule):

    def __init__(self, name, directory):
        super().__init__(name, directory)

        self.fc = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.fc.weight, 0)

    def forward(self, x):
        return self.fc(x)
