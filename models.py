import torch
import torch.nn as nn
import torch.nn.functional as F


class SOrthConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_mode='zeros'):
        '''
        Conv1d with a method for stepping towards semi-orthongonality
        http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(SOrthConv, self).__init__()

        kwargs = {'bias': False}
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=False, padding_mode=padding_mode)
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        return x

    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.conv)
            self.conv.weight.copy_(M)

    def reset_parameters(self):
        # Standard dev of M init values is inverse of sqrt of num cols
        nn.init._no_grad_normal_(self.conv.weight, 0.,
                                 self.get_M_shape(self.conv.weight)[1]**-0.5)

    def orth_error(self):
        return self.get_semi_orth_error(self.conv).item()

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.99
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])

    @staticmethod
    def get_semi_orth_error(conv1dlayer):
        with torch.no_grad():
            orig_shape = conv1dlayer.weight.shape
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            scale2 = torch.sqrt(trace_PP/trace_P) ** 2
            update = P - (torch.matrix_power(P, 0) * scale2)
            return torch.norm(update, p='fro')

class SharedDimScaleDropout(nn.Module):
    def __init__(self, alpha: float = 0.5, dim=1):
        '''
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        '''
        super(SharedDimScaleDropout, self).__init__()
        if alpha > 0.5 or alpha < 0:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha
        self.dim = dim
        self.register_buffer('mask', torch.tensor(0.))
         
    def forward(self, X):
        if self.training:
            if self.alpha != 0.:
                # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
                tied_mask_shape = list(X.shape)
                tied_mask_shape[self.dim] = 1
                repeats = [1 if i != self.dim else X.shape[self.dim] for i in range(len(X.shape))]
                return X * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*self.alpha, 1 + 2*self.alpha).repeat(repeats)
                # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return X

class FTDNNLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, bottleneck_dim, context_size=2, dilations=None, paddings=None, alpha=0.0):
        '''
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(FTDNNLayer, self).__init__()
        paddings = [1,1,1] if not paddings else paddings
        dilations = [2,2,2] if not dilations else dilations
        kwargs = {'bias':False}
        self.factor1 = nn.Conv1d(in_dim, bottleneck_dim, context_size, padding=paddings[0], dilation=dilations[0], **kwargs)
        self.factor2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, context_size, padding=paddings[1], dilation=dilations[1], **kwargs)
        self.factor3 = nn.Conv1d(bottleneck_dim, out_dim, context_size, padding=paddings[2], dilation=dilations[2], **kwargs)
        self.reset_parameters()
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = SharedDimScaleDropout(alpha=alpha, dim=1)
        
    def forward(self, x):
        ''' input (batch_size, seq_len, in_dim) '''
        assert (x.shape[-1] == self.factor1.weight.shape[1])
        x = self.factor1(x.transpose(1,2))
        x = self.factor2(x)
        x = self.factor3(x)
        x = self.nl(x)
        x = self.bn(x).transpose(1,2)
        x = self.dropout(x)
        return x
    
    def step_semi_orth(self):
        with torch.no_grad():
            factor1_M = self.get_semi_orth_weight(self.factor1)
            factor2_M = self.get_semi_orth_weight(self.factor2)
            self.factor1.weight.copy_(factor1_M)
            self.factor2.weight.copy_(factor2_M)
    
    def reset_parameters(self):
        # Standard dev of M init values is inverse of sqrt of num cols
        nn.init._no_grad_normal_(self.factor1.weight, 0., self.get_M_shape(self.factor1.weight)[1]**-0.5)
        nn.init._no_grad_normal_(self.factor2.weight, 0., self.get_M_shape(self.factor2.weight)[1]**-0.5)
        
    def orth_error(self):
        factor1_err = self.get_semi_orth_error(self.factor1).item()
        factor2_err = self.get_semi_orth_error(self.factor2).item()
        return factor1_err + factor2_err

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]: # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.999
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])

    @staticmethod
    def get_semi_orth_error(conv1dlayer):
        with torch.no_grad():
            orig_shape = conv1dlayer.weight.shape
            M = conv1dlayer.weight.reshape(orig_shape[0], orig_shape[1]*orig_shape[2])
            mshape = M.shape
            if mshape[0] > mshape[1]: # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)
            scale2 = torch.sqrt(trace_PP/trace_P) ** 2
            update = P - (torch.matrix_power(P, 0) * scale2)
            return torch.norm(update, p='fro')

class DenseReLU(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(DenseReLU, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nl = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.nl(x)
        if len(x.shape) > 2:
            x = self.bn(x.transpose(1,2)).transpose(1,2)
        else:
            x = self.bn(x)
        return x

class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x
