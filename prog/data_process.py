import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from graphset import *
from scipy.sparse import coo_matrix

def CalculateGraphFeat(feat_mat,adj_list):
    use_molecular_graph = True
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    if use_molecular_graph==True:
        for i in range(len(adj_list)):
            nodes = adj_list[i]
            for each in nodes:
                adj_mat[i,int(each)] = 1
        assert np.allclose(adj_mat,adj_mat.T)
    else:
        adj_mat = adj_mat + np.eye(len(adj_list))
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat,adj_index]

def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat,adj_list,_ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat,adj_list)
    return drug_data

def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0:int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask

def process(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_move, nb_celllines, nb_drugs):
    cellineid = list(set([item[0] for item in data_move]));cellineid.sort()
    pubmedid = list(set([item[1] for item in data_move]));pubmedid.sort()
    cellmap = list(zip(cellineid,list(range(len(cellineid)))))
    pubmedmap = list(zip(pubmedid,list(range(len(cellineid),len(cellineid)+len(pubmedid)))))
    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0]==j[0]] for i in data_move])
    pubmed_num = np.squeeze([[j[1] for j in pubmedmap if i[1]==j[0]] for i in data_move])
    IC_num = np.squeeze([i[2] for i in data_move])
    new = np.vstack((cellline_num,pubmed_num,IC_num)).T
    new = new[new[:,2].argsort()]
    cellid=[item[0] for item in cellmap]
    gexpr_feature=gexpr_feature.loc[cellid]
    mutation_feature=mutation_feature.loc[cellid]
    methylation_feature=methylation_feature.loc[cellid]
    pubid=[item[0] for item in pubmedmap]
    drug_feature=pd.DataFrame(drug_feature).T
    drug_feature=drug_feature.loc[pubid]
    atom_shape=drug_feature[0][0].shape[-1]
##----drug
    drug_data = FeatureExtract(drug_feature)

    mutation = torch.from_numpy(np.array(mutation_feature,dtype='float32'))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature,dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation_feature,dtype='float32'))
    ###loader
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data),collate_fn=collate,batch_size=nb_drugs,shuffle=False)
    cellline_set = Data.DataLoader(dataset=Data.TensorDataset(mutation,gexpr,methylation),batch_size=nb_celllines,shuffle=False)
    use_independent_testset=False
    if(use_independent_testset == True):
        edge_mask = cmask(len(new), 0.1, 666)
        train = new[edge_mask][:, 0:3]
        test = new[~edge_mask][:, 0:3]
    else:
        testedge_mask = cmask(len(new), 0.1, 666)
        validation = new[testedge_mask][:, 0:3]
        vali_mask = cmask(len(validation), 0.2, 66)
        train = validation[vali_mask][:, 0:3]
        test = validation[~vali_mask][:, 0:3]
    train[:, 1] -= nb_celllines
    test[:, 1] -= nb_celllines
    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])),
                            shape=(nb_celllines, nb_drugs)).toarray()
    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])),
                           shape=(nb_celllines, nb_drugs)).toarray()
    train_mask = torch.from_numpy(train_mask).view(-1)
    test_mask = torch.from_numpy(test_mask).view(-1)
    if (use_independent_testset == True):
        pos_edge = new[new[:, 2] == 1, 0:2]
        neg_edge = new[new[:, 2] == -1, 0:2]
    else:
        pos_edge = validation[validation[:, 2] == 1, 0:2]
        neg_edge = validation[validation[:, 2] == -1, 0:2]
    pos_edge[:, 1] -= nb_celllines
    neg_edge[:, 1] -= nb_celllines
    label_pos = coo_matrix((np.ones(pos_edge.shape[0]), (pos_edge[:, 0], pos_edge[:, 1])),
                           shape=(nb_celllines, nb_drugs)).toarray()
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)
    if (use_independent_testset == True):
        train_edge = new[edge_mask]
    else:
        train_edge = validation[vali_mask]
    train_edge = np.vstack((train_edge, train_edge[:, [1, 0, 2]]))

    return drug_set,cellline_set,train_edge,label_pos,train_mask,test_mask,atom_shape


