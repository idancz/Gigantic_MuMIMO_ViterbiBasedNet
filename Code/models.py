import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from Code.channel.channel_estimation import estimate_channel
from Code.channel.modulator import BPSKModulator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias): 
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, activation=True):
        super(ConvUpsampleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.activation = activation
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        if self.activation:
            out = self.relu(out)
        return out


#ResNet with 5/2 kernel/padding
class BasicResidualUnit(nn.Module):
    """Basic residual unit for building ResNet with 1D input"""
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicResidualUnit, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        
        input_tensor_copy = input_tensor
        
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu1(output_tensor)
        
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm2(output_tensor)
        
        if self.downsample is not None:
            self.downsample(output_tensor)
        
        output_tensor += input_tensor_copy
        #output_tensor = self.relu2(output_tensor)
        return output_tensor
    
class BasicResidualStack(nn.Module):
    """Basic residual stack, made from 1x1 conv, ResUnit, ResUnit, MaxPool"""
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicResidualStack, self).__init__()
        self.downsample = downsample
        
        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,)
        self.basic_unit_1 = BasicResidualUnit(in_channels=out_channels, out_channels=out_channels)
        self.basic_unit_2 = BasicResidualUnit(in_channels=out_channels, out_channels=out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, input_tensor):
        
        output_tensor = self.conv1x1(input_tensor)
        output_tensor = self.basic_unit_1(output_tensor)
        output_tensor = self.basic_unit_2(output_tensor)
        output_tensor = self.maxpool(output_tensor)
        
        return output_tensor


class Residual1DModel(nn.Module):
    def __init__(self, in_size, in_channels, block_sizes, num_classes):
        super(Residual1DModel, self).__init__()
        self.n_classes = num_classes
        self.block_sizes = block_sizes
        
        self.in_size = in_size
        num_features = in_size[1]
        
        self.basic_stack_1 = BasicResidualStack(in_channels, block_sizes[0], )
        num_features /= 2
            
        self.basic_stack_2 = BasicResidualStack(block_sizes[0], block_sizes[1], )
        num_features /= 2
        
        self.basic_stack_3 = BasicResidualStack(block_sizes[1], block_sizes[2], )
        num_features /= 2
        
        if len(self.block_sizes) >= 4:
            self.basic_stack_4 = BasicResidualStack(block_sizes[2], block_sizes[3], )
            num_features /= 2
        
        if len(self.block_sizes) >= 5:
            self.basic_stack_5 = BasicResidualStack(block_sizes[3], block_sizes[4], )
            num_features /= 2
            
        if len(self.block_sizes) >= 6:
            self.basic_stack_6 = BasicResidualStack(block_sizes[4], block_sizes[5], )
            num_features /= 2
                
        self.fc_1 = nn.Linear(in_features=int(block_sizes[-1] * num_features), out_features=128, )
        self.selu_1 = nn.SELU(inplace=True)
        self.a_dropout_1 = nn.AlphaDropout(.1)
        
        self.fc_2 = nn.Linear(128, 128)
        self.selu_2 = nn.SELU(inplace=True)
        self.a_dropout_2 = nn.AlphaDropout(.1)
        
        self.fc_last = nn.Linear(128, num_classes)       

    def forward(self, input_tensor):
        
        output_tensor = self.basic_stack_1(input_tensor)        
        
        #print("Here", output_tensor.shape)
        
        output_tensor = self.basic_stack_2(output_tensor)        
        output_tensor = self.basic_stack_3(output_tensor)
        if len(self.block_sizes) >= 4: output_tensor = self.basic_stack_4(output_tensor)
        if len(self.block_sizes) >= 5: output_tensor = self.basic_stack_5(output_tensor)
        if len(self.block_sizes) >= 6: output_tensor = self.basic_stack_6(output_tensor)
            
        output_tensor = self.fc_1(torch.flatten(output_tensor, 1))
        output_tensor = self.selu_1(output_tensor)
        output_tensor = self.a_dropout_1(output_tensor)
        
        output_tensor = self.fc_2(output_tensor)
        output_tensor = self.selu_2(output_tensor)
        output_tensor = self.a_dropout_2(output_tensor)
        
        output_tensor = self.fc_last(output_tensor)  
        
        return output_tensor


class EncoderConv(nn.Module):
    def __init__(self, input_samples: int, dim: int, n_classes: int, debug=False):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_samples
        dropout = 0
        self.debug = debug

        self.encoder = nn.Sequential(
            ConvBlock(1, dim // 2, 5, 2, False),
            # nn.MaxPool1d(2),
            ConvBlock(dim // 2, dim // 2, 5, 2, False),  # added
            ConvBlock(dim // 2, dim, 5, 2, False),
            # nn.MaxPool1d(2),
            ConvBlock(dim, dim, 5, 2, False),
            ConvBlock(dim, dim, 5, 2, False)
        )

        ff_out_size = int(dim * input_samples // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * int(input_samples), ff_out_size),  # input_samples/4, 1024
            nn.BatchNorm1d(ff_out_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # 0.2
            nn.Linear(ff_out_size, n_classes), )

    def forward(self, input_):
        z = self.encoder(input_.reshape(-1, 1, self.input_size))

        if self.debug: print(z.shape)
        y = self.classifier(z)
        return y


class EncoderDecoder(nn.Module):
    def __init__(self, input_samples: int, dim: int, n_classes: int, debug=False):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_samples
        self.debug = debug

        self.encoder = nn.Sequential(
            ConvBlock(1, dim // 8, 5, 2, False),
            ConvBlock(dim // 8, dim // 4, 5, 2, False),
            ConvBlock(dim//4, dim // 2, 5, 2, False),
            ConvBlock(dim // 2, dim, 5, 2, False),
            ConvBlock(dim, dim, 5, 2, False),
            ConvBlock(dim, dim, 5, 2, False), )

        self.decoder = nn.Sequential(
            ConvUpsampleBlock(dim, dim // 2, 5, 2, False),
            ConvUpsampleBlock(dim // 2, dim // 2, 5, 2, True),
            ConvUpsampleBlock(dim // 2, dim // 4, 5, 2, True),
            ConvUpsampleBlock(dim // 4, dim // 4, 5, 2, True),
            ConvUpsampleBlock(dim // 4, 1, 5, 2, True, activation=False), )

        self.fc_out = nn.Linear(input_samples, n_classes)

    def forward(self, input_):
        z = self.encoder(input_.reshape(-1, 1, self.input_size))

        if self.debug: print(z.shape)

        recon = self.decoder(z)

        if self.debug: print(recon.shape)

        y = self.fc_out(recon).permute(1, 0, 2)

        return y


class Basic1DResidualNet(nn.Module):

    def __init__(self, in_size, out_size, in_channels, block_sizes, num_classes, dropout=0.1):
        super(Basic1DResidualNet, self).__init__()
        self.n_classes = num_classes
        self.block_sizes = block_sizes
        self.input_size = in_size[1]
        self.dropout = dropout

        self.in_size = in_size
        num_features = in_size[1]

        self.basic_stack_1 = BasicResidualStack(in_channels, block_sizes[0], )
        num_features //= 2

        self.basic_stack_2 = BasicResidualStack(block_sizes[0], block_sizes[1], )
        num_features //= 2

        # self.basic_stack_3 = BasicResidualStack(block_sizes[1], block_sizes[2], )
        # num_features //= 2

        if len(self.block_sizes) >= 4:
            self.basic_stack_4 = BasicResidualStack(block_sizes[2], block_sizes[3], )
            num_features /= 2

        if len(self.block_sizes) >= 5:
            self.basic_stack_5 = BasicResidualStack(block_sizes[3], block_sizes[4], )
            num_features /= 2

        if len(self.block_sizes) >= 6:
            self.basic_stack_6 = BasicResidualStack(block_sizes[4], block_sizes[5], )
            num_features /= 2

        self.fc_1 = nn.Linear(in_features=int(block_sizes[-1] * num_features), out_features=out_size)
        self.selu_1 = nn.SELU(inplace=True)
        self.a_dropout_1 = nn.AlphaDropout(dropout)

        self.fc_2 = nn.Linear(out_size, out_size)
        self.selu_2 = nn.SELU(inplace=True)
        self.a_dropout_2 = nn.AlphaDropout(dropout)

        self.fc_last = nn.Linear(out_size, num_classes)

    def forward(self, input_tensor):

        output_tensor = self.basic_stack_1(input_tensor.reshape(-1, 1, self.input_size))
        output_tensor = self.basic_stack_2(output_tensor)
        # output_tensor = self.basic_stack_3(output_tensor)

        if len(self.block_sizes) >= 4: output_tensor = self.basic_stack_4(output_tensor)
        if len(self.block_sizes) >= 5: output_tensor = self.basic_stack_5(output_tensor)
        if len(self.block_sizes) >= 6: output_tensor = self.basic_stack_6(output_tensor)

        output_tensor = self.fc_1(torch.flatten(output_tensor, 1))
        output_tensor = self.selu_1(output_tensor)
        output_tensor = self.a_dropout_1(output_tensor)

        output_tensor = self.fc_2(output_tensor)
        output_tensor = self.selu_2(output_tensor)
        output_tensor = self.a_dropout_2(output_tensor)

        output_tensor = self.fc_last(output_tensor)

        return output_tensor


class ADNN(nn.Module):
    def __init__(self, input_size: int, dim: int,  n_classes: int):
        super(ADNN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            ConvBlock(1, dim // 8, 5, 2, False),
            # nn.MaxPool1d(2),
            ConvBlock(dim // 8, dim // 4, 5, 2, False),
            # nn.MaxPool1d(2),
            ConvBlock(dim // 4, dim //2,  5, 2, False),
            # nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            ConvUpsampleBlock(dim // 2, dim, 5, 2, False),
            # nn.Upsample(scale_factor=2),
            ConvUpsampleBlock(dim, dim //2, 5, 2, False),
            # nn.Upsample(scale_factor=2),
            ConvUpsampleBlock(dim // 2, dim // 4, 5, 2, False),
            # nn.Upsample(scale_factor=2),
            ConvUpsampleBlock(dim // 4, dim // 8, 5, 2, False, activation=True),
        )


        self.dnn = nn.Sequential( # permute to filter dim
            nn.Flatten(),
            # nn.Sigmoid(),
            nn.Linear(input_size*(dim // 8), n_classes),  # input_samples/4, 1024
        )
        self.fc = nn.Linear(dim, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_):
        batch_size, transmission_length = input_.size(0), input_.size(1)

        z = self.encoder(input_.reshape(-1, 1, self.input_size))
        y = self.decoder(z)
        out = self.dnn(y.unsqueeze(1))
        return out.reshape(batch_size, transmission_length,  self.n_classes)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, padding_mode='circular'):
        super(Conv2dBlock, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ADNN2D(nn.Module):
    def __init__(self, input_size: int, dim: int,  n_classes: int):
        super(ADNN2D, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        self.expand = nn.Sequential(ConvBlock(1, 2, kernel_size=3, padding=1, bias=False))
        self.encoder = nn.Sequential(
            Conv2dBlock(1, dim//8, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.MaxPool2d(2),
            Conv2dBlock(dim//8, dim // 4, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.MaxPool2d(2),
            Conv2dBlock(dim // 4, dim // 2, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(dim // 2, dim, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim // 2, dim // 4, kernel_size=(3, 5), padding=(1, 2), bias=False),
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim // 4, dim // 8, kernel_size=(3, 5), padding=(1, 2), bias=False),
        )

        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*input_size*(dim//8), dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
            # nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*input_size, n_classes)
        )

    def forward(self, inp):
        x = self.expand(inp.reshape(-1, 1, self.input_size))
        z = self.encoder(x.unsqueeze(1))
        y = self.decoder(z)
        out = self.dnn(y)
        # y = y.permute(0, 2, 3, 1)
        # y_bar = self.dnn(y).permute(0, 3, 1, 2)
        # out = self.fc(y_bar)
        return out


class DecoderConv(nn.Module):
    def __init__(self, input_samples: int, dim: int, n_classes: int, debug=False):
        super().__init__()
        self.input_size = input_samples
        self.n_classes = n_classes
        scale = 2
        n_up_layers = 2
        self.debug = debug

        self.decoder = nn.Sequential(
        ConvUpsampleBlock(1, dim//2, 5, 2, False),
        nn.Upsample(scale_factor=scale),
        ConvUpsampleBlock(dim//2, dim//2, 5, 2, True),
        nn.Upsample(scale_factor=scale),
        ConvUpsampleBlock(dim//2, dim//4, 5, 2, True),
        ConvUpsampleBlock(dim//4, dim, 5, 2, True, activation=False),)

        ff_out_size = int(dim * input_samples)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_up_layers * scale * dim * int(input_samples), ff_out_size),  # input_samples/4, 1024
            nn.BatchNorm1d(ff_out_size),
            nn.ReLU(),
            nn.Linear(ff_out_size, n_classes),
        )

    def forward(self, input_):
        z = self.decoder(input_.reshape(-1, 1, self.input_size))

        if self.debug: print(z.shape)
        y = self.classifier(z)
        return y


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(LSTM, self).__init__()

        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, sequence_y: torch.Tensor) -> torch.Tensor:
        ## LSTM Model Starts ##
        batch_size, transmission_length = sequence_y.size(0), sequence_y.size(1)

        # Set initial states
        h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_n = torch.zeros(self.num_layers, batch_size,  self.hidden_size).to(device)

        # Forward propagate LSTM - lstm_out: tensor of shape (batch_size, seq_length, hidden_size*2)
        lstm_out = torch.zeros(batch_size, transmission_length, self.hidden_size).to(device)
        for i in range(batch_size):
            lstm_out[i], temp = self.lstm(sequence_y[i].unsqueeze(0),
                                          (h_n[:, i].unsqueeze(1).contiguous(), c_n[:, i].unsqueeze(1).contiguous()))

        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = self.fc(lstm_out.reshape(-1, self.hidden_size)).reshape(batch_size, transmission_length,  self.n_classes)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, input_size, num_conv_channels):
        super(ResidualBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(input_size)
        self.conv1d_1 = nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2, bias=False)
        self.layer_norm_2 = nn.LayerNorm(input_size)
        self.conv1d_2 = nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        z = self.layer_norm_1(inputs)  # .permute(0, 2, 1)
        z = self.relu(z)
        z = self.conv1d_1(z)
        z = self.layer_norm_2(z)
        z = self.relu(z)
        z = self.conv1d_2(z)  # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs
        return z


class SionnaNeuralReceiver(nn.Module):
    def __init__(self, input_size, n_input_channels, n_output_channels, n_classes):
        super(SionnaNeuralReceiver, self).__init__()
        k = 2
        self.input_size = input_size
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes

        self.input_layer = nn.Sequential(nn.Linear(input_size, k*input_size, bias=False))

        self._input_conv = nn.Conv1d(n_input_channels, n_output_channels, kernel_size=5, padding=2, bias=False)
        # Residual blocks
        self._res_block_1 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_2 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_3 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_4 = ResidualBlock(k*input_size, n_output_channels)
        # Output conv
        self._output_conv = nn.Conv1d(n_output_channels, n_classes, kernel_size=5, padding=2, bias=False)

        self.fc_out = nn.Linear(k*input_size*n_classes, n_classes, bias=False)

    def forward(self, inputs):
        batch_size, transmission_length = inputs.size(0), inputs.size(1)
        # Input conv
        z = self._input_conv(self.input_layer(inputs.reshape(-1, self.n_input_channels, self.input_size)))
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)
        z = self.fc_out(z.reshape(batch_size, transmission_length, -1))
        return z.reshape(batch_size, transmission_length, -1)


class SionnaSkip(nn.Module):
    def __init__(self, input_size, n_input_channels, n_output_channels, n_classes):
        super(SionnaSkip, self).__init__()
        k = 2
        self.input_size = input_size
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes

        self.input_layer = nn.Sequential(nn.Linear(input_size, k*input_size, bias=False))

        self._input_conv = nn.Conv1d(n_input_channels, n_output_channels, kernel_size=5, padding=2, bias=False)
        # Residual blocks
        self._res_block_1 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_2 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_3 = ResidualBlock(k*input_size, n_output_channels)
        self._res_block_4 = ResidualBlock(k*input_size, n_output_channels)
        # Output conv
        self._output_conv = nn.Conv1d(n_output_channels, n_classes, kernel_size=5, padding=2, bias=False)

        self.fc_out = nn.Linear(k*input_size*n_classes, n_classes, bias=False)

    def forward(self, inputs):
        batch_size, transmission_length = inputs.size(0), inputs.size(1)

        # Input conv
        z = self._input_conv(self.input_layer(inputs.reshape(-1, self.n_input_channels, self.input_size)))
        # Residual blocks
        z1 = self._res_block_1(z)
        z2 = self._res_block_2(z+z1)
        z3 = self._res_block_3(z+z1+z2)
        z4 = self._res_block_4(z+z1+z2+z3)
        # Output conv
        out = self._output_conv(z4)
        out = self.fc_out(out.reshape(batch_size, transmission_length, -1))
        return out.reshape(batch_size, transmission_length, -1)


class ConvTranspose1D(nn.Module):
    def __init__(self, input_size: int, dim: int,  n_classes: int):
        super(ConvTranspose1D, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.ConvTranspose1d(1, dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm1d(dim),
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.ConvTranspose1d(dim, dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim//2),
            nn.ConvTranspose1d(dim//2, dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * dim//2 * input_size, n_classes)
        )

    def forward(self, input_):
        batch_size, transmission_length = input_.size(0), input_.size(1)
        x = self.encoder(input_.reshape(-1, 1, self.input_size))
        out = self.fc(x)
        return out.reshape(batch_size, transmission_length, -1)


class FullyConnected(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FullyConnected, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        n_input = input_size
        n_dense_1 = 64
        n_dense_2 = 64
        n_dense_3 = 64
        n_out = n_classes
        self.net = nn.Sequential(

                    # first hidden layer:
                    nn.Linear(n_input, n_dense_1),
                    nn.ReLU(),

                    # second hidden layer:
                    nn.Linear(n_dense_1, n_dense_2),
                    nn.ReLU(),

                    # third hidden layer:
                    nn.Linear(n_dense_2, n_dense_3),
                    nn.ReLU(),
                    # nn.Dropout(),

                    # output layer:
                    nn.Linear(n_dense_3, n_out),
                )

    def forward(self, inputs):
        batch_size, transmission_length = inputs.size(0), inputs.size(1)
        out = self.net(inputs.reshape(batch_size, transmission_length, self.input_size))
        return out.reshape(batch_size, transmission_length, -1)


class SionnaViterbiPlus(nn.Module):
    def __init__(self, input_size, n_input_channels, n_output_channels, n_classes):
        super(SionnaViterbiPlus, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.sionna = SionnaNeuralReceiver(input_size, n_input_channels, n_output_channels, n_classes)
        self.viterbinet = ViterbiNet(input_size, n_classes)
        self.fc = nn.Sequential(
            nn.Linear(2*n_classes, n_classes, bias=False)
        )

    def forward(self, input_):
        batch_size, transmission_length = input_.size(0), input_.size(1)

        out = self.sionna(input_)
        vout = self.viterbinet(input_)
        combined = torch.cat((out, vout), dim=-1)
        priors = self.fc(combined)
        return priors.reshape(batch_size, transmission_length, self.n_classes)


class SionnaViterbiAdd(nn.Module):
    def __init__(self, input_size, n_input_channels, n_output_channels, n_classes):
        super(SionnaViterbiAdd, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.sionna = SionnaNeuralReceiver(input_size, n_input_channels, n_output_channels, n_classes)
        self.viterbinet = ViterbiNet(input_size, n_classes)

    def forward(self, input_):
        batch_size, transmission_length = input_.size(0), input_.size(1)

        out = self.sionna(input_)
        vout = self.viterbinet(input_)
        priors = out + vout
        return priors.reshape(batch_size, transmission_length, self.n_classes)


####################################### ViterbiNet ##############################################
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 50

class ViterbiNet(nn.Module):
    """
    This implements the ViterbiNet decoder by an NN on each stage
    """
    def __init__(self, input_size, n_classes: int):

        super(ViterbiNet, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN1_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, self.n_classes)
        ).to(device)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet model
        :param y: input values, size [batch_size,transmission_length]
        :returns the estimated priors [batch_size,transmission_length,n_classes]
        """
        # compute priors
        priors = self.net(y.reshape(-1, 1)).reshape(y.size(0), y.size(1), self.n_classes)
        return priors


####################################### Transformers ##############################################


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        if N > 1:
            self.norm2 = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class ECC_Transformer(nn.Module):
    def __init__(self, input_size, n_dim, n_heads, n_layers, n_classes, dropout=0):
        super(ECC_Transformer, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        cpy = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, n_dim)
        ff = PositionwiseFeedForward(n_dim, n_dim*4, dropout)
        self.input_layer = nn.Linear(input_size, n_dim, bias=False)
        self.transformer_encoder = Encoder(EncoderLayer(n_dim, cpy(attn), cpy(ff), dropout), n_layers)

        self.fc = nn.Linear(n_dim, n_classes)

    def generate_square_subsequent_mask(self, size: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).type(torch.bool).to(device)


    def forward(self, input_):
        batch_size, transmission_length = input_.size(0), input_.size(1)

        x = self.input_layer(input_)
        src_mask = self.generate_square_subsequent_mask(transmission_length)
        y = self.transformer_encoder(x, src_mask)  # src_mask = None
        out = self.fc(y)
        return out.reshape(batch_size, transmission_length, self.n_classes)


class TRANSFORMER(nn.Module):
    def __init__(self, input_size, n_dim, n_heads, num_layers, ff_dim, n_classes, dropout=0):
        super(TRANSFORMER, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.n_dim = n_dim
        dropout = dropout

        t_encoder = nn.TransformerEncoderLayer(
            d_model=n_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=ff_dim,
            # activation=torch.sigmoid,
            norm_first=True,
            # layer_norm_eps=1e-6
        ).to(device)
        # Stack the encoder layer n times in nn.TransformerDecoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=t_encoder,
            num_layers=num_layers,
            norm=None
        ).to(device)

        self.position_layer = nn.Sequential(
            nn.Linear(1, n_dim, bias=False),
            # nn.Sigmoid(),
            # nn.Linear(4*dim, dim)
        ).to(device)

        self.input_encoder_layer = nn.Sequential(nn.Linear(input_size, n_dim, bias=False),
                                                ).to(device)

        self.output_decoder_layer = nn.Sequential(nn.ReLU(),
                                                  nn.Linear(n_dim, n_classes, bias=False)
                                                  ).to(device)

    def create_src_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_size, transmission_length = y.size(0), y.size(1)

        x = self.input_encoder_layer(y)  # input (batch, 136, 4) --> out (batch, 136, 128)
        # mask = self.create_src_mask(transmission_length)
        x = self.transformer_encoder(x)  # input (batch, 136, 128) --> out (batch, 136, 128)
        out = self.output_decoder_layer(x)  # input (batch, 136, 128) --> out (batch, 136, 2)  .permute(1,0,2)
        out = out.reshape(batch_size, transmission_length, self.n_classes)
        # out = out[:, 1:].reshape(batch_size, transmission_length-1, self.n_classes)
        return out


class ConvTRANSFORMER(nn.Module):
    def __init__(self, input_size, dim, heads, n_layers, ff_dim, n_classes, dropout=0):
        super(ConvTRANSFORMER, self).__init__()
        self.n_classes = ff_dim
        self.input_size = input_size
        self.n_dim = dim

        self.position_layer = nn.Linear(1, 2*dim*input_size, bias=False)

        dropout = dropout
        num_layers = n_layers
        t_encoder = nn.TransformerEncoderLayer(
            d_model=2*dim*input_size,
            nhead=heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=ff_dim,
            # activation=torch.sigmoid,
            norm_first=True,
            # layer_norm_eps=1e-6
        ).to(device)
        # Stack the encoder layer n times in nn.TransformerDecoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=t_encoder,
            num_layers=num_layers,
            norm=None
        ).to(device)

        self.input_encoder_layer = nn.Sequential(
            nn.ConvTranspose1d(1, dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm1d(dim//2),
            nn.ConvTranspose1d(dim // 2, dim // 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
            ConvBlock(dim // 2, dim, 5, 2, False),
            ConvBlock(dim, dim, 5, 2, False),
            # nn.AvgPool1d(2),
            # nn.Upsample(scale_factor=2),

        ).to(device)


        self.output_decoder_layer = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
        ).to(device)

        self.fc = nn.Sequential(
             nn.Linear(int(2*dim*self.input_size), self.n_classes, bias=False),

        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.n_classes)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_size, transmission_length = y.size(0), y.size(1)

        x = self.input_encoder_layer(y.reshape(-1, 1, self.input_size))
        # x = self.adp(x.reshape(batch_size, transmission_length, -1))
        out = self.transformer_encoder(x.reshape(batch_size, transmission_length, -1))
        # out = self.output_decoder_layer(z)
        out = self.fc(out.reshape(batch_size, transmission_length, -1))
        out = out.reshape(batch_size, transmission_length, self.n_classes)

        # x = self.transformer_encoder(x.reshape(batch_size, transmission_length, -1))  # input (batch, 136, 128) --> out (batch, 136, 128)
        # out = self.output_decoder_layer(x)  # input (batch, 136, 128) --> out (batch, 136, 2)  .permute(1,0,2)
        # out = out.reshape(batch_size, transmission_length, self.n_classes)
        # out = out[:, 1:].reshape(batch_size, transmission_length-1, self.n_classes)
        return out



#######################################  Statistical Viterbi  ##############################################

class ClassicViterbi(nn.Module):
    """
    This module implements the classic statistical Viterbi Algorithm detector
    """

    def __init__(self,
                 n_classes: int,
                 memory_length: int,
                 gamma: float,
                 val_words: int,
                 channel_type: str,
                 noisy_est_var: float,
                 fading: bool,
                 fading_taps_type: int,
                 channel_coefficients: str):

        super(ClassicViterbi, self).__init__()
        self.memory_length = memory_length
        self.gamma = gamma
        self.val_words = val_words
        self.n_classes = n_classes
        self.channel_type = channel_type
        self.noisy_est_var = noisy_est_var
        self.fading = fading
        self.fading_taps_type = fading_taps_type
        self.channel_coefficients = channel_coefficients
        self.input_size = 1
        self.count = 0

    def compute_state_priors(self, h: np.ndarray) -> torch.Tensor:
        all_states_decimal = np.arange(self.n_classes).astype(np.uint8).reshape(-1, 1)
        all_states_binary = np.unpackbits(all_states_decimal, axis=1).astype(int)
        if self.channel_type == 'ISI_AWGN':
            all_states_symbols = BPSKModulator.modulate(all_states_binary[:, -self.memory_length:])
        else:
            raise Exception('No such channel defined!!!')
        state_priors = np.dot(all_states_symbols, h.T)
        return torch.Tensor(state_priors).to(device)

    def compute_likelihood_priors(self, y: torch.Tensor, count: int = None):
        # estimate channel per word (only changes between the h's if fading is True)
        h = np.concatenate([estimate_channel(self.memory_length, self.gamma, noisy_est_var=self.noisy_est_var,
                                             fading=self.fading, index=index, fading_taps_type=self.fading_taps_type,
                                             channel_coefficients=self.channel_coefficients) for index in range(self.val_words)], axis=0)
        if count is not None:
            h = h[count].reshape(1, -1)
        # compute priors
        state_priors = self.compute_state_priors(h)
        if self.channel_type == 'ISI_AWGN':
            priors = y.unsqueeze(dim=2) - state_priors.T.repeat(
                repeats=[y.shape[0] // state_priors.shape[1], 1]).unsqueeze(
                dim=1)
            # to llr representation
            priors = priors ** 2 / 2 - math.log(math.sqrt(2 * math.pi))
        else:
            raise Exception('No such channel defined!!!')
        return priors

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns priors
        """
        # compute transition likelihood priors
        priors = self.compute_likelihood_priors(y.reshape(1, -1), self.count)
        self.count += 1
        return -priors


