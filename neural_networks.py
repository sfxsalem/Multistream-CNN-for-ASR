
import ast
import math
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FTDNNLayer, DenseReLU, TDNN, StatsPool


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



    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options["cnn_N_filt"].split(",")))

        self.cnn_len_filt = list(map(int, options["cnn_len_filt"].split(",")))
        self.cnn_max_pool_len = list(map(int, options["cnn_max_pool_len"].split(",")))

        self.cnn_act = options["cnn_act"].split(",")
        self.cnn_drop = list(map(float, options["cnn_drop"].split(",")))

        self.cnn_use_laynorm = list(map(strtobool, options["cnn_use_laynorm"].split(",")))
        self.cnn_use_batchnorm = list(map(strtobool, options["cnn_use_batchnorm"].split(",")))
        self.cnn_use_laynorm_inp = strtobool(options["cnn_use_laynorm_inp"])
        self.cnn_use_batchnorm_inp = strtobool(options["cnn_use_batchnorm_inp"])

        self.N_cnn_lay = len(self.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])])
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = x.view(batch, -1)
        return x



    def __init__( self, options, inp_dim=440):

        super(ETDNN, self).__init__()

        self.embed_features = 3480

        self.dropout_p = 0.0
        self.batch_norm = True
        tdnn_kwargs = {'dropout_p':self.dropout_p, 'batch_norm':self.batch_norm}
        self.nl = nn.LeakyReLU()

        self.frame1 = TDNN(input_dim=inp_dim, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame6 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame7 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame8 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame9 = TDNN(input_dim=512, output_dim=512*3, context_size=1, dilation=1, **tdnn_kwargs)

        self.tdnn_list = nn.Sequential(self.frame1, self.frame2, self.frame3, self.frame4, self.frame5, self.frame6, self.frame7, self.frame8, self.frame9)
        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(512*6, self.embed_features) 
        self.out_dim = 3480

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch, 1, seq_len)
        # x = self.frame1(x)
        # print("SHAPE 1: ",x.shape)
        x = self.tdnn_list(x)
        # x = self.frame2(x)
        # print("SHAPE 2: ",x.shape)
        x = self.statspool(x)
        # print("SHAPE 3: ",x.shape)
        x = self.fc_embed(x)
        # print("SHAPE 4: ",x.shape)
        return x

class FTDNN(nn.Module):
    
    def __init__(self, inp_dim=440):

        super(FTDNN, self).__init__()
        
        self.input_dim = inp_dim
        self.layer01 = TDNN(input_dim=self.input_dim, output_dim=512, context_size=5, padding=2)
        self.layer02 = FTDNNLayer(512, 1024, 256, context_size=1)
        self.layer03 = FTDNNLayer(1024, 1024, 256, context_size=1, dilations=[1,1,1], paddings=[0,0,0])
        self.layer04 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3,3,2], paddings=[2,1,1])
        self.layer05 = FTDNNLayer(2048, 1024, 256, context_size=1, dilations=[1,1,1], paddings=[0,0,0])
        self.layer06 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3,3,2], paddings=[2,1,1])
        self.layer07 = FTDNNLayer(3072, 1024, 256, context_size=2, dilations=[3,3,2], paddings=[2,1,1])
        self.layer08 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3,3,2], paddings=[2,1,1])
        self.layer09 = FTDNNLayer(3072, 1024, 256, context_size=1, dilations=[1,1,1], paddings=[0,0,0])
        self.layer10 = DenseReLU(1024, 2048)
        
        self.layer11 = StatsPool()
        
        self.layer12 = DenseReLU(4096, 440)
        # self.act = nn.LogSoftmax(dim=1)
        self.out_dim = 440
        
        
    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch, 1, seq_len)
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        skip_5 = torch.cat([x_4, x_3], dim=-1)
        x = self.layer05(skip_5)
        x_6 = self.layer06(x)
        skip_7 = torch.cat([x_6, x_4, x_2], dim=-1)
        x = self.layer07(skip_7)
        x_8 = self.layer08(x)
        skip_9 = torch.cat([x_8, x_6, x_4], dim=-1)
        x = self.layer09(skip_9)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        # x = self.act(x)
        x = x.reshape(batch, -1)
        return x

    def step_mbs_layers(self):
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

class MultiStreamBlock(nn.Module):

    def __init__(self, options, inp_dim=440, out_dim= 3480):
        super(MultiStreamBlock, self).__init__()

        ## Reading parameters
        
        # Input Dimension
        self.input_dim = inp_dim
        self.out_dim = out_dim

        # MultiStreamBlock Parameters
        self.num_streams = int(options["num_streams"])
        self.tdnnf_bttlnck = int(options["tdnnf_bttlnck"])
        self.tdnnf_out = list(map(int, options["tdnnf_out"].split(",")))
        self.tdnnf_dilations = list(ast.literal_eval(options['tdnnf_dilations']))
        self.streams = nn.ModuleList()

        # Single Stream Setup
        self.sstream = FTDNN()

        # Multi Stream Setup
        for i in range(self.num_streams):
            stream = TDNNF(self.input_dim, self.out_dim, self.tdnnf_bttlnck, self.tdnnf_out[i], self.tdnnf_dilations[i])
            self.streams.append(stream)
            
        self.dense = nn.Linear(self.out_dim*self.num_streams, self.out_dim, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        
        encs = []
        x = self.sstream(x)
        for stream in self.streams:
            encs += [stream(x)]
        enc = torch.cat(encs, dim=1)
        h = self.dense(enc)
        h = self.dropout(self.bn(self.relu(h)))
        return h
    
    def step_mbs_layers(self):
        for layer in self.children():
            if isinstance(layer, TDNNF):
                layer.step_ftdnn_layers()


class TDNNF(nn.Module):
    def __init__(self, inp_dim=440, out_dim= 3480, bttlnck_dim = 256, layerout_dim = 1024, dilations = [1, 1, 1]):
        super(TDNNF, self).__init__()

        ## Reading parameters
        
        # Input Dimension
        self.input_dim = inp_dim
        self.out_dim = out_dim
        self.bttlnck_dim = bttlnck_dim
        self.layerout_dim = layerout_dim
        self.dilations = dilations

        # F-TDNN Layers
        self.layer01 = FTDNNLayer(self.input_dim, self.layerout_dim, self.bttlnck_dim, context_size=1,  dilations=self.dilations)
        self.layer02 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        self.layer03 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        self.layer04 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer05 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer06 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer07 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer08 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer09 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer10 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer11 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer12 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer13 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer14 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=2, dilations=self.dilations)
        # self.layer15 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer16 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)
        # self.layer17 = FTDNNLayer(self.layerout_dim, self.layerout_dim, self.bttlnck_dim, context_size=1, dilations=self.dilations)

        self.layer18 = DenseReLU(self.layerout_dim, self.layerout_dim*2)
        self.layer19 = StatsPool()
        self.layer20 = DenseReLU(self.layerout_dim*4,self.out_dim)

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch, 1, seq_len)
        x_1 = self.layer01(x)
        x_2 = self.layer02(x_1)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        # x_5 = self.layer05(x_4)
        # # skip_6 = torch.cat([x_5, x_3, x_1], dim=-1)
        # x_6 = self.layer06(x_5)
        # x_7 = self.layer07(x_6)
        # # skip_8 = torch.cat([x_7, x_5, x_3], dim=-1)
        # x_8 = self.layer08(x_7)
        # x_9 = self.layer09(x_8)
        # # skip_10 = torch.cat([x_9, x_7], dim=-1)
        # x_10 = self.layer10(x_9)
        # x_11 = self.layer11(x_10)
        # x_12 = self.layer12(x_11)
        # skip_13 = torch.cat([x_12, x_9, x_7],dim=-1)
        # x_13 = self.layer13(x_12)
        # x_14 = self.layer14(x_13)
        # x_15 = self.layer15(x_14)
        # # skip_16 = torch.cat([x_13, x_11, x_9], dim=-1)
        # x_16 = self.layer16(x_15)
        # x_17 = self.layer17(x_16)
        x_20 = self.layer20(self.layer19(self.layer18(x_4)))
        x = x_20.reshape(batch, -1)
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
   