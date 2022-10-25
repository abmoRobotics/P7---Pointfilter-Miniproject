from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=3):#, patch_nums=500, sym_op='max'):
        super(Encoder, self).__init__()
        #self.patch_nums = patch_nums
        #self.sym_op = sym_op
        self.input_dim = input_dim
        self.convs = []
        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.convs.append(self.conv1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.convs.append(self.conv2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.convs.append(self.conv3)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.convs.append(self.conv4)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1)
        self.convs.append(self.conv5)

        self.bns = []
        self.bn1 = nn.BatchNorm1d(64)
        self.bns.append(self.bn1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bns.append(self.bn2)
        self.bn3 = nn.BatchNorm1d(256)
        self.bns.append(self.bn3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bns.append(self.bn4)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bns.append(self.bn5)

        self.activate = nn.ReLU()

    def forward(self, x):
        for c, bn in self.convs, self.bns:
            x = self.activate(bn(c(x)))
        if self.sym_op == 'sum':
            x = torch.sum(x, dim=-1)
        else:
            x, index = torch.max(x, dim=-1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x

class Pointfilternet(nn.Module):
    def __init__(self, input_dim=3, patch_nums=500, sym_op='max'):
        super(Pointfilternet, self).__init__()

        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.encoder = Encoder(self.input_dim)#, self.patch_nums, self.sym_op)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    model = Pointfilternet()#.cuda()
    print(model)

