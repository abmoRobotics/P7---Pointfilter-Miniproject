from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=3, sym_op='max'):
        super(Encoder, self).__init__()
        self.sym_op = sym_op
        self.input_dim = input_dim
        self.conv_layers = []
        self.add_conv_layer(self.input_dim, 64, 1)
        self.add_conv_layer(64, 128, 1)
        self.add_conv_layer(128, 256, 1)
        self.add_conv_layer(256, 512, 1)
        self.add_conv_layer(512, 1024, 1)
        self.convs = nn.ModuleList(self.conv_layers)
        
        self.bns = []
        self.add_batch_norm1d(64)
        self.add_batch_norm1d(128)
        self.add_batch_norm1d(256)
        self.add_batch_norm1d(512)
        self.add_batch_norm1d(1024)
        self.bns_params = nn.ModuleList(self.bns)

        self.activate = nn.ReLU()


    def add_conv_layer(self, num_input, num_output, kernel_size):
        self.conv_layers.append(nn.Conv1d(num_input, num_output, kernel_size=kernel_size))


    def add_batch_norm1d(self, num_features):
        self.bns.append(nn.BatchNorm1d(num_features))


    def forward(self, x):
        op = torch.max if self.sym_op == 'max' else torch.sum
        print(op)
        for c in range(len(self.conv_layers)):
            x = self.activate(self.bns[c](self.conv_layers[c](x)))
        x = op(x, dim=-1)
        return x[0] if self.sym_op == 'max' else x 


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.lns = []
        self.add_linear_transform(1024, 512)
        self.add_linear_transform(512, 256)
        self.add_linear_transform(256, 3)
        self.lns_params = nn.ModuleList(self.lns)

        self.bns = []
        self.add_batch_norm1d(512)
        self.add_batch_norm1d(256)
        self.bns_params = nn.ModuleList(self.bns)

    
    def add_batch_norm1d(self, num_features):
        self.bns.append(nn.BatchNorm1d(num_features))


    def add_linear_transform(self, input_features, output_features):
        self.lns.append(nn.Linear(input_features, output_features))


    def forward(self, x):
        for i in range(len(self.lns)):
            if i == len(self.lns)-1:
                x = torch.tanh(self.lns[i](x))
            else:
                x = F.relu(self.bns[i](self.lns[i](x)))
        return x


class Pointfilternet(nn.Module):
    def __init__(self, input_dim=3):
        super(Pointfilternet, self).__init__()
        self.input_dim = input_dim

        self.encoder = Encoder(self.input_dim)
        self.decoder = Decoder()

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


if __name__ == '__main__':
    model = Pointfilternet()
    print(model)

