import torch
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


class MultimodalAutoencoder(nn.Module):
    def __init__(self, f_in, layers, dropout, hidden_units, f_embedding, sw_l_acc, sw_l_gyr, sw_r_acc, sw_r_gyr,
                 eb_acc, eb_gyr, sp_acc, sp_gyr, skel, return_embeddings=False):
        super(MultimodalAutoencoder, self).__init__()
        
        self.sw_l_acc = sw_l_acc
        self.sw_l_gyr = sw_l_gyr
        self.sw_r_acc = sw_r_acc
        self.sw_r_gyr = sw_r_gyr
        self.eb_acc = eb_acc
        self.eb_gyr = eb_gyr
        self.sp_acc = sp_acc
        self.sp_gyr = sp_gyr
        self.skel = skel
        
        self.return_embeddings = return_embeddings
        self.layers = layers
        self.f_in = f_in
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        if layers == 2:
            self.enc_fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.enc_fc2 = nn.Linear(hidden_units, f_embedding, bias=True)
        elif layers == 3:
            self.enc_fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.enc_fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.enc_fc3 = nn.Linear(hidden_units, f_embedding, bias=True)
        
        # Decoder
        if layers == 2:
            self.dec_fc1 = nn.Linear(f_embedding, hidden_units, bias=True)
            self.dec_fc2 = nn.Linear(hidden_units, f_in, bias=True)
        elif layers == 3:
            self.dec_fc1 = nn.Linear(f_embedding, hidden_units, bias=True)
            self.dec_fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.dec_fc3 = nn.Linear(hidden_units, f_in, bias=True)

    def forward(self, skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g):        
        self.sw_l_acc.set_decode_mode(False)
        self.sw_l_gyr.set_decode_mode(False)
        self.sw_r_acc.set_decode_mode(False)
        self.sw_r_gyr.set_decode_mode(False)
        self.eb_acc.set_decode_mode(False)
        self.eb_gyr.set_decode_mode(False)
        self.sp_acc.set_decode_mode(False)
        self.sp_gyr.set_decode_mode(False)
        self.skel.set_decode_mode(False)
        
        skel = self.skel(skel)
        
        eb_l_a = self.eb_acc(eb_l_a)
        eb_l_g = self.eb_gyr(eb_l_g)

        sp_r_a = self.sp_acc(sp_r_a)
        sp_r_g = self.sp_gyr(sp_r_g)
        
        sw_l_a = self.sw_l_acc(sw_l_a)
        sw_l_g = self.sw_l_gyr(sw_l_g)
        sw_r_a = self.sw_r_acc(sw_r_a)
        sw_r_g = self.sw_r_gyr(sw_r_g)
        
        skel_size = skel.size()
        eb_l_a_size, eb_l_g_size = eb_l_a.size(), eb_l_g.size()
        sp_r_a_size, sp_r_g_size = sp_r_a.size(), sp_r_g.size()
        sw_l_a_size, sw_l_g_size, sw_r_a_size, sw_r_g_size = sw_l_a.size(), sw_l_g.size(), sw_r_a.size(), sw_r_g.size()
        
        skel = skel.view(skel.size()[0], -1)
        eb_l_a = eb_l_a.view(eb_l_a.size()[0], -1)
        eb_l_g = eb_l_g.view(eb_l_g.size()[0], -1)
        sp_r_a = sp_r_a.view(sp_r_a.size()[0], -1)        
        sp_r_g = sp_r_g.view(sp_r_g.size()[0], -1)
        sw_l_a = sw_l_a.view(sw_l_a.size()[0], -1)
        sw_l_g = sw_l_g.view(sw_l_g.size()[0], -1)
        sw_r_a = sw_r_a.view(sw_r_a.size()[0], -1)
        sw_r_g = sw_r_g.view(sw_r_g.size()[0], -1)
        
        cat_inds = [skel.size()[1], eb_l_a.size()[1], eb_l_g.size()[1], sp_r_a.size()[1], sp_r_g.size()[1],
                    sw_l_a.size()[1], sw_l_g.size()[1], sw_r_a.size()[1], sw_r_g.size()[1]]
        
        x = torch.cat((skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g), 1)
        
        if self.layers == 2:
            x = self.enc_fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc2(x)
            if self.return_embeddings:
                return x
            
            x = self.relu(x)
            x = self.dropout(x)
            x = self.dec_fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.dec_fc2(x)
            
        elif self.layers == 3:            
            x = self.enc_fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc3(x)
            if self.return_embeddings:
                return x
            
            x = self.relu(x)
            x = self.dropout(x)
            x = self.dec_fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.dec_fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.dec_fc3(x)
        
        skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g = torch.split(x, cat_inds, dim=1)
        
        skel = skel.view(skel_size)
        eb_l_a = eb_l_a.view(eb_l_a_size)
        eb_l_g = eb_l_g.view(eb_l_g_size)
        sp_r_a = sp_r_a.view(sp_r_a_size)
        sp_r_g = sp_r_g.view(sp_r_g_size)
        sw_l_a = sw_l_a.view(sw_l_a_size)
        sw_l_g = sw_l_g.view(sw_l_g_size)
        sw_r_a = sw_r_a.view(sw_r_a_size)
        sw_r_g = sw_r_g.view(sw_r_g_size)
        
        self.sw_l_acc.set_decode_mode(True)
        self.sw_l_gyr.set_decode_mode(True)
        self.sw_r_acc.set_decode_mode(True)
        self.sw_r_gyr.set_decode_mode(True)
        self.eb_acc.set_decode_mode(True)
        self.eb_gyr.set_decode_mode(True)
        self.sp_acc.set_decode_mode(True)
        self.sp_gyr.set_decode_mode(True)
        self.skel.set_decode_mode(True)
        
        skel = self.skel(skel)
        
        eb_l_a = self.eb_acc(eb_l_a)
        eb_l_g = self.eb_gyr(eb_l_g)
        
        sp_r_a = self.sp_acc(sp_r_a)
        sp_r_g = self.sp_gyr(sp_r_g)
        
        sw_l_a = self.sw_l_acc(sw_l_a)
        sw_l_g = self.sw_l_gyr(sw_l_g)
        sw_r_a = self.sw_r_acc(sw_r_a)
        sw_r_g = self.sw_r_gyr(sw_r_g)
        
        return skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g
