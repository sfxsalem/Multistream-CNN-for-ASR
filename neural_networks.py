
import ast
import math
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FTDNNLayer, DenseReLU, StatsPool

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        
def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initialized like this, but not used in forward!

class MLP(nn.Module):
    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()

        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options["dnn_lay"].split(",")))
        self.dnn_drop = list(map(float, options["dnn_drop"].split(",")))
        self.dnn_use_batchnorm = list(map(strtobool, options["dnn_use_batchnorm"].split(",")))
        self.dnn_use_laynorm = list(map(strtobool, options["dnn_use_laynorm"].split(",")))
        self.dnn_use_laynorm_inp = strtobool(options["dnn_use_laynorm_inp"])
        self.dnn_use_batchnorm_inp = strtobool(options["dnn_use_batchnorm_inp"])
        self.dnn_act = options["dnn_act"].split(",")

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):

            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        return x

class MultiStreamBlock(nn.Module):

    def __init__(self, options, inp_dim=440, out_dim=3416):
        super(MultiStreamBlock, self).__init__()

        ## Reading parameters
        self.in_dim = inp_dim
        self.out_dim = out_dim

        # SingleStream Parameters
        self.ss_in_dim = 40  #default:inp_dim, trying other 
        self.ss_out_dim = int(options["ss_out_dim"]) #512
        self.ss_bttlnck_dim = int(options["ss_bttlnck_dim"]) #128

        # MultiStream Parameters
        self.ms_in_dim = int(options["ms_in_dim"]) #512
        self.ms_out_dim = int(options["ms_out_dim"]) #512
        self.ms_bttlnck_dim = int(options["ms_bttlnck_dim"]) #128

        self.ms_context_size = int(options["ms_context_size"]) or list(ast.literal_eval(options["ms_context_size"]))
        self.ms_dilations = list(ast.literal_eval(options["ms_dilations"]))
        self.ms_paddings = list(ast.literal_eval(options["ms_paddings"]))

        self.num_layers = int(options["num_layers"]) #17

        self.num_streams = len(self.ms_dilations)
        self.streams = nn.ModuleList()

        # Single Stream Setup
        self.sstream = SingleStream()

        # Multi Stream Setup
        for i in range(self.num_streams):
            stream = MultiStream(self.ms_in_dim, self.ms_out_dim, self.ms_bttlnck_dim, self.ms_context_size, self.ms_dilations, self.ms_paddings, self.num_layers)
            self.streams.append(stream)
            
        
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.ms_out_dim)
        self.dropout = nn.Dropout(p=0.15)

        self.dense01 = DenseReLU(self.ms_out_dim, self.ms_out_dim * 2)
        self.sp01 = StatsPool()
        self.dense02 = DenseReLU(self.ms_out_dim * 4, self.out_dim)


    def forward(self, x):
        # Reshape the input to (batch_size, seq_len, in_dim)
        if x.ndim == 2:
            batch = x.shape[0]
            seq_len = x.shape[1]
            x = x.view(batch, -1, 40)

        # single stream
        x = self.sstream(x)

        # multi stream
        encs = []
        for stream in self.streams:
            encs += [stream(x)]
        enc = torch.cat(encs, dim=1)

        h = self.dropout(self.bn(self.relu(enc.transpose(1, 2)))).transpose(1, 2)
        h = self.dense01(h)
        h = self.sp01(h)
        h = self.dense02(h)

        #h = self.dropout(self.bn(self.relu(x.transpose(1, 2)))).transpose(1, 2)
        #h = self.dense01(h)
        #h = self.sp01(h)
        #h = self.dense02(h)
        return h
    
    def step_mbs_layers(self):
        for streams in self.children():
            if isinstance(streams, SingleStream):
                streams.step_ftdnn_layers()
            elif isinstance(streams, nn.ModuleList):
                for stream in streams:
                    if isinstance(stream, MultiStream):
                        stream.step_ftdnn_layers()

class SingleStream(nn.Module):
    
    def __init__(self, inp_dim=40, out_dim=512, bttlnck_dim=128, context_size=2, dilations=[2, 2, 2], paddings=[1, 1, 1]):

        super(SingleStream, self).__init__()
        
        self.input_dim = inp_dim
        self.layer01 = FTDNNLayer(self.input_dim, out_dim, bttlnck_dim, context_size, dilations, paddings)
        self.layer02 = FTDNNLayer(out_dim, out_dim, bttlnck_dim, context_size, dilations, paddings)
        self.layer03 = FTDNNLayer(out_dim, out_dim, bttlnck_dim, context_size, dilations, paddings)
        self.layer04 = FTDNNLayer(out_dim, out_dim, bttlnck_dim, context_size, dilations, paddings)
        self.layer05 = FTDNNLayer(out_dim, out_dim, bttlnck_dim, context_size, dilations, paddings)
        
        
        
    def forward(self, x):

        #batch = x.shape[0]
        #seq_len = x.shape[1]
        #x = x.view(batch, 1, seq_len)
        x = self.layer01(x)
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.layer05(x)

        #x = x.reshape(batch, -1)
        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors

class MultiStream(nn.Module):
    def __init__(self, inp_dim, out_dim= 512, bttlnck_dim = 128, context_size=2, dilations = [1, 1, 1], paddings=[1, 1, 1], num_layers=17):
        super(MultiStream, self).__init__()

        # Input Dimension
        self.input_dim = inp_dim
        self.out_dim = out_dim
        self.bttlnck_dim = bttlnck_dim
        self.context_size = context_size
        self.dilations = dilations
        self.paddings = paddings
        self.num_layers = num_layers

        # Layers Module
        self.layers = nn.ModuleList()

        # Input Layer
        self.layer01 = FTDNNLayer(self.input_dim, self.out_dim, self.bttlnck_dim, self.context_size, self.dilations, self.paddings)
        self.layers.append(self.layer01)

        # Further Layers
        for i in range(self.num_layers):
            layer = FTDNNLayer(self.out_dim, self.out_dim, self.bttlnck_dim, self.context_size, self.dilations, self.paddings)
            self.layers.append(layer)

    def forward(self, x):

        #batch = x.shape[0]
        #seq_len = x.shape[1]
        #x = x.view(batch, 1, seq_len)

        for layer in self.layers:
            x = layer(x)
        
        #x = x.reshape(batch, -1)
        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors
   
