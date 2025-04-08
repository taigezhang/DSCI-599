# code/model.py

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout=0.2):
        super().__init__()
        self.sage1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.sage2 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.sage3 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.out = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.dropout = dropout

    def forward(self, g, x):
        x = F.relu(self.sage1(g, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage2(g, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage3(g, x))
        return self.out(g, x)
