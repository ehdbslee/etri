# ===== Torch Library ===== #
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    def __init__(self, in_dim, emb_dim, n_layer):
        super(Embedding, self).__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_layer = n_layer

        # ===== Create Linear Layers ===== #
        self.fc1 = nn.Linear(self.in_dim, self.emb_dim)

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.bns.append(nn.BatchNorm1d(self.emb_dim))

        self.fc2 = nn.Linear(self.emb_dim, self.emb_dim)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x = self.act(self.linears[i](x))
            x = self.bns[i](x)
        x = self.fc2(x)
        return x


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act, use_bn, use_xavier):
        super(Net, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act
        self.use_bn = use_bn
        self.use_xavier = use_xavier


        # ===== Create Linear Layers ===== #
        self.fc1 = nn.Linear(self.in_dim, self.hid_dim)

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
            if self.use_bn: self.bns.append(nn.BatchNorm1d(self.hid_dim))

        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)


        # ====== Create Activation Function ====== #
        if self.act == 'relu': self.act = nn.ReLU()
        elif self.act == 'leakyrelu': self.act = nn.LeakyReLU()
        elif self.act == 'tanh': self.act = nn.Tanh()
        elif self.act == 'sigmoid': self.act = nn.Sigmoid()
        else: raise ValueError('no valid activation function')


        # ====== Create Regularization Layer ======= #
        if self.use_xavier: self.xavier_init()


    def forward(self, x):
        x = self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x = self.act(self.linears[i](x))
            if self.use_bn==True: x = self.bns[i](x)
        x = self.fc2(x)
        return x


    def xavier_init(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0.01)