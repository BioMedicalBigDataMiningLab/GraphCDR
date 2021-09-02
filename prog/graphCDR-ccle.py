import torch
import pandas as pd
import rdkit
from rdkit import Chem
import deepchem as dc
import time
from model import *
from data_process import process
import argparse
from my_utiils import *
from data_load import dataload

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--alph', dest='alph', type=float, default=0.30, help='')
parser.add_argument('--beta', dest='beta', type=float, default=0.30, help='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='')
parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=256, help='')
parser.add_argument('--output_channels', dest='output_channels', type=int, default=100, help='')
args = parser.parse_args()
start_time = time.time()
#--------cellline_feature_input
Genomic_mutation_file = '../data/Celline/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = '../data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
mutation_feature = pd.read_csv(Genomic_mutation_file,sep=',',header=0,index_col=[0])
mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]
#--------drug_feature_input
drug='../data/CCLE/CCLE_smiles.csv'
drug=pd.read_csv(drug, sep=',',header=0)
drug_feature = {}
featurizer = dc.feat.ConvMolFeaturizer()
for tup in zip(drug['pubchem'], drug['isosmiles']):
    mol=Chem.MolFromSmiles(tup[1])
    X = featurizer.featurize(mol)
    drug_feature[str(tup[0])]=[X[0].get_atom_features(),X[0].get_adjacency_list(),1]
#--------response_input
response='../data/CCLE/CCLE_response.csv'
datar=pd.read_csv(response, sep=',',header=0)
data_idx = []
thred=0.8
for tup in zip(datar['DepMap_ID'],datar['pubchem'],datar['Z_SCORE']):
    t=1 if tup[2]>thred else -1
    data_idx.append((tup[0],str(tup[1]),t))

data_s=sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
data_back=[];data_move=[]
data_idx1 = [[i[0],i[1]] for i in data_s]
for i,k in zip(data_idx1,data_s):
    if i not in data_back:
        data_back.append(i)
        data_move.append(k)
nb_celllines = len(set([item[0] for item in data_move]))
nb_drugs = len(set([item[1] for item in data_move]))
print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_move),nb_celllines,nb_drugs))

drug_set,cellline_set,train_edge,label_pos,train_mask,test_mask,atom_shape = process(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_move, nb_celllines, nb_drugs)

model = DeepGraphInfomax(hidden_channels=args.hidden_channels, encoder=Encoder(args.output_channels, args.hidden_channels), summary=Summary(args.output_channels, args.hidden_channels),
                 feat=model_feature(atom_shape,gexpr_feature.shape[-1],methylation_feature.shape[-1],args.output_channels),index=nb_celllines)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss()

def train():
    model.train()
    loss_temp=0
    for batch, (drug,cell) in enumerate(zip(drug_set,cellline_set)):
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, pos_adj=model(drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge)
        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = myloss(pos_adj[train_mask],label_pos[train_mask])
        loss=(1-args.alph-args.beta)*pos_loss + args.alph*dgi_pos + args.beta*dgi_neg
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()
    print('train loss: ', str(round(loss_temp, 4)))

def test():
    model.eval()
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj=model(drug.x, drug.edge_index, drug.batch,cell[0], cell[1], cell[2], train_edge)
            loss_temp = myloss(pre_adj[test_mask],label_pos[test_mask])
        yp=pre_adj[test_mask].detach().numpy()
        ytest=label_pos[test_mask].detach().numpy()
        AUC, AUPR, F1, ACC =metrics_graph(ytest,yp)
        print('test loss: ', str(round(loss_temp.item(), 4)))
        print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
              '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
    return AUC, AUPR, F1, ACC

#------main
final_AUC = 0;final_AUPR = 0;final_F1 = 0;final_ACC = 0
for epoch in range(args.epoch):
    print('\nepoch: ' + str(epoch))
    train()
    AUC, AUPR, F1, ACC = test()
    if (AUC > final_AUC):
        final_AUC = AUC;final_AUPR = AUPR;final_F1 = F1;final_ACC = ACC
elapsed = time.time() - start_time
print('---------------------------------------')
print('Elapsed time: ', round(elapsed, 4))
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
      '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(round(final_ACC, 4)))
print('---------------------------------------')