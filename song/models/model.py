import torch
import torch.nn as nn

class HandPoseModel(nn.Module):
    def __init__(self):
        super(HandPoseModel, self).__init__()
        # 입력 42개, 은닉층 64개, 출력 12개(동작 종류)
        self.layer1 = nn.Linear(42, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 12)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out