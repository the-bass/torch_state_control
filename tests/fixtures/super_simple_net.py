import torch.nn as nn


class SuperSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.fc.weight, 0)

    def forward(self, x):
        return self.fc(x)
