import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, SGConv, global_mean_pool
from torch.nn import Parameter
from my_utiils import *
EPS = 1e-15
class NodeAttribute(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeAttribute, self).__init__()
        torch.manual_seed(0)
        # -------drug_layer(4 layers)
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        self.conv1 = SGConv(gcn_layer, units_list[0], add_self_loops=False)
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        self.graph_conv = []
        self.graph_bn = []
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGConv(units_list[i], units_list[i + 1], add_self_loops=False))
            self.graph_bn.append(nn.BatchNorm1d((units_list[i + 1])))
        self.conv_end = SGConv(units_list[-1], output, add_self_loops=False)
        self.batch_end = nn.BatchNorm1d(output)
        # -------gexp_layer
        self.fc_gexp1 = nn.Linear(dim_gexp, 256)
        self.batch_gexp1 = nn.BatchNorm1d(256)
        self.fc_gexp2 = nn.Linear(256, output)
        # -------methy_layer
        self.fc_methy1 = nn.Linear(dim_methy, 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, output)
        # -------mut_layer
        self.cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(2010, output)
        # ------Concatenate_celline
        self.fcat = nn.Linear(300, output)
        self.batchc = nn.BatchNorm1d(100)
        # self.prelu = nn.PReLU(output)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_conv1(x_drug)
        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)
        if self.use_GMP:
            x_drug = gmp(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # -----mutation_train  #genomic mutation feature
        if self.use_mutation:
            x_mutation = torch.tanh(self.cov1(mutation_data))
            x_mutation = F.max_pool2d(x_mutation, (1, 5))
            x_mutation = F.relu(self.cov2(x_mutation))
            x_mutation = F.max_pool2d(x_mutation, (1, 10))
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = F.relu(self.fc_mut(x_mutation))
            # x_mutation = torch.dropout(x_mutation, 0.1, train=False)

        # ----gexpr_train #gexp feature
        if self.use_gexpr:
            x_gexpr = torch.tanh(self.fc_gexp1(gexpr_data))
            x_gexpr = self.batch_gexp1(x_gexpr)
            # x_gexpr = torch.dropout(x_gexpr,0.1, train=False)
            x_gexpr = F.relu(self.fc_gexp2(x_gexpr))

        # ----methylation_train
        if self.use_methylation:
            x_methylation = torch.tanh(self.fc_methy1(methylation_data))
            x_methylation = self.batch_methy1(x_methylation)
            # x_methylation = torch.dropout(x_methylation, 0.1, train=False)
            x_methylation = F.relu(self.fc_methy2(x_methylation))

        # ------Concatenate
        if self.use_gexpr==False:
            x_cell = torch.cat((x_mutation, x_methylation), 1)
        elif self.use_mutation==False:
            x_cell = torch.cat((x_gexpr, x_methylation), 1)
        elif self.use_methylation == False:
            x_cell = torch.cat((x_mutation, x_gexpr), 1)
        else:
            x_cell = torch.cat((x_mutation, x_gexpr, x_methylation), 1)
        x_cell = F.relu(self.fcat(x_cell))
        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batchc(x_all)
        return x_all

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu1 = nn.PReLU(hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=True)
        # self.prelu2 = nn.PReLU(hidden_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        # x = self.conv2(x, edge_index)
        # x = self.prelu2(x)
        return x

class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 1)

    def forward(self, xo, xn):
        m = self.fc1(torch.cat((xo, xn), 1))
        m = torch.tanh(torch.squeeze(m))
        m = torch.exp(m) / (torch.exp(m)).sum()
        x = torch.matmul(m, xn)
        return x


class DeepGraphInfomax(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(100, 10)
        self.fd = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T)
        neg_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2].T)
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        cellpos = pos_z[:self.index, ]; drugpos = pos_z[self.index:, ]
        cellfea = self.fc(feature[:self.index, ]); drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea); drugfea = torch.sigmoid(drugfea)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
