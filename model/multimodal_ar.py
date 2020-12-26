import torch.nn as nn
    

class MultimodalFcClassifier(nn.Module):
    def __init__(self, f_in, num_classes, multimodal_ae_model, dropout=0.0, layers=3, hidden_units=100):
        super(MultimodalFcClassifier, self).__init__()
        self.f_in = f_in
        self.multimodal_ae_model = multimodal_ae_model
        self.layers = layers
        
        if layers == 2:
            self.fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.fc2 = nn.Linear(hidden_units, num_classes, bias=True)
        elif layers == 3:
            self.fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.fc3 = nn.Linear(hidden_units, num_classes, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g):
        x = self.multimodal_ae_model(skel, eb_l_a, eb_l_g, sp_r_a, sp_r_g, sw_l_a, sw_l_g, sw_r_a, sw_r_g)
        
        if self.layers == 2:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        elif self.layers == 3:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
        
        return x
