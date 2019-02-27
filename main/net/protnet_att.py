import torch.nn as nn
import torch
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, inp, out, ms=(4,4), ds=1):
        super(GLU, self).__init__()
        fs = (3,3)
        ps = (1,1)
        self.ms = ms
        self.cnn_lin = nn.Conv2d(inp, out, fs, dilation=ds, padding=ps, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.mp = nn.MaxPool2d(ms)

    def forward(self, x):
        out = F.relu(self.bn(self.cnn_lin(x)))
        return self.mp(out)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.model_name = 'ProtNet_att'
        ks = 32*4
        self.G1 = GLU(   1, ks*1)
        self.G2 = GLU(ks*1, ks*2)
        self.G3 = GLU(ks*2, ks*3, (1,1))
        self.att = nn.Conv2d(ks*3, ks*3, (3,3), padding=(1,1), bias=False)

    def nn_att(self, inp, att):
        att_out = att(inp)  # (130, 384, 8, 10)
        att_out = F.softmax(att_out.view(att_out.size(0), att_out.size(1), -1), dim=2)  # (130, 384, 80)

        att_sc = att_out.sum(1).view(att_out.size(0), 1, att_out.size(2))  # (130, 1, 80)
        att_sc = att_sc.div(att_out.size(1))
        att_sc = att_sc.repeat(1, att_out.size(1), 1)  # (130, 384, 80)
        return att_sc

    def forward(self, x, xavg, xstd, n=5, m=5):
        zx = (x - xavg) / xstd  # (130, 1, 128, 160)
        G1 = self.G1(zx)  # (130, 128, 32, 40)
        G2 = self.G2(G1)  # (130, 256, 8, 10)
        G3 = self.G3(G2)  # (130, 384, 8, 10)
        att = self.nn_att(G3, self.att)
        embed = G3.view(G3.size(0), G3.size(1), -1) * att  # (130, 384, 80)
        embed = embed.sum(-1)  # (130, 384)
        embed = embed.view(-1, n * m + 1, embed.size(1))  # (5, 26, 384)
        # query -> (5, 5, 384)
        query = embed[:, -1].view(-1, 1, embed.size(2)).repeat(1, n, 1)
        # support -> (5, 5, 384)
        support = embed[:, :-1].view(embed.size(0), n, m, -1).mean(2)
        sim = torch.pow(query - support, 2).sum(-1)  # (5, 5)
        return -sim
