import torch.nn as nn


class DropoutNeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(DropoutNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.input_dropout = nn.Dropout(p=0.2)
        self.hidden_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.input_dropout(x)
        out = self.l1(out)
        out = self.relu(out)
        out = self.hidden_dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.hidden_dropout(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
