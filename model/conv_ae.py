import torch.nn as nn

    
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, layers, input_size, grouped, dim, input_ch=3, kernel_size=5, kernel_stride=3,
                 return_embeddings=False, decode_only=False):
        super(ConvAutoencoder, self).__init__()
        
        self.dim = dim
        self.return_embeddings = return_embeddings
        self.decode_only = decode_only
        self.pool_ind = None
        self.pool_size = None
        self.k_s = kernel_stride
        self.layers = layers
        self.relu = nn.ReLU()
        
        if dim == 1:
            conv_f = nn.Conv1d
            conv_t_f = nn.ConvTranspose1d
            if layers == 3 and kernel_stride == 2:
                out_padding = [0, 0, 1]
            else: 
                out_padding = [0, 0, 0]
            pool_f = nn.MaxPool1d
            unpool_f = nn.MaxUnpool1d
            self.interp = Interpolate(input_size, 'linear')
        else:
            conv_f = nn.Conv2d
            conv_t_f = nn.ConvTranspose2d
            if layers == 3 and kernel_stride[0] == 2:
                out_padding = [(1, 0), (0, 0), (1, 0)]
            else: 
                out_padding = [(0, 0), (0, 0), (0, 0)]
            pool_f = nn.MaxPool2d
            unpool_f = nn.MaxUnpool2d
            self.interp = Interpolate(input_size, 'bilinear')
        
        padding = (kernel_size-1)//2
        ch_mult = 3//input_ch
        
        # Encoder
        self.conv1 = conv_f(input_ch, 9//ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                            padding=padding, groups=grouped[0])
        if self.layers > 1:
            self.conv2 = conv_f(9//ch_mult, 15//ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, groups=grouped[1])
        if self.layers > 2:  
            self.conv3 = conv_f(15//ch_mult, 24//ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, groups=grouped[2])
        self.pool = pool_f(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # Decoder
        self.unpool = unpool_f(kernel_size=2, stride=2, padding=0)
        if self.layers > 2:
            self.deconv1 = conv_t_f(24//ch_mult, 15//ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                    padding=padding, output_padding=out_padding[0], groups=grouped[-1])
        if self.layers > 1:
            self.deconv2 = conv_t_f(15//ch_mult, 9//ch_mult, kernel_size=kernel_size, stride=kernel_stride,
                                    padding=padding, output_padding=out_padding[1], groups=grouped[-1])
        self.deconv3 = conv_t_f(9//ch_mult, input_ch, kernel_size=kernel_size, stride=kernel_stride,
                                padding=padding, output_padding=out_padding[2], groups=grouped[-1])

    def forward(self, x, pool_ind=None):
        
        if not self.decode_only:
            # Encoder
            x = self.conv1(x)
            x = self.relu(x)

            if self.layers > 1:
                x = self.conv2(x)
                x = self.relu(x)

            if self.layers > 2:
                x = self.conv3(x)
                x = self.relu(x)

            self.pool_size = x.size()
            x, pool_ind = self.pool(x)
            self.pool_ind = pool_ind

            if self.return_embeddings:
                return x
        
        # Decoder
        if self.dim == 2:  # Bug in PyTorch 1.0.1. requires this [https://github.com/pytorch/pytorch/issues/16486]
            x = self.unpool(x, self.pool_ind, output_size=self.pool_size)
        else:
            x = self.unpool(x, self.pool_ind)
        
        if self.layers > 2:
            x = self.deconv1(x)
            x = self.relu(x)
        
        if self.layers > 1:
            x = self.deconv2(x)
            x = self.relu(x)
        
        x = self.deconv3(x)

        if self.dim == 1:
            if self.layers < 3 or self.k_s == 3:
                x = self.interp(x)
        elif self.dim == 2:
            if self.layers < 3 or self.k_s[0] == 3:
                x = self.interp(x)
        
        return x
    
    def set_decode_mode(self, val):
        self.decode_only = val
