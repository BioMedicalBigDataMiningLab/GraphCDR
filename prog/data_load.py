import numpy as np
import csv
import pandas as pd
import hickle as hkl
import os
def dataload(Drug_info_file, IC50_threds_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file,
             Cancer_response_exp_file, Gene_expression_file, Methylation_file):
#-----drug_dataload------#
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
    drug2thred={}
    for line in open(IC50_threds_file).readlines()[1:]:
        drug2thred[str(line.split('\t')[0])]=float(line.strip().split('\t')[1])
    '''
    IC50threds = pd.read_csv(IC50_threds_file, sep=',',header=0)
    drug2dict  = dict(zip(IC50threds['pubchem'],IC50threds['IC50']))
    IC50key=[]; IC50value=[]
    for key, value in drug2dict.items():
        key=str(key)
        IC50key.append(key)
        IC50value.append(value)
    drug2thred = dict(zip(IC50key,IC50value))
    '''
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())

#-----cell line_dataload------#
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label
    mutation_feature = pd.read_csv(Genomic_mutation_file,sep=',',header=0,index_col=[0])
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
    assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])

#-----drug_cell line_pairs dataload------#
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    data_idx = []
    use_thred=True
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    if use_thred:
                        if pubchem_id in drug2thred.keys():
                            binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else -1
                            data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline]))
                    else:
                        binary_IC50 = 1 if ln_IC50 < -2 else -1
                        data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline]))
  #----eliminate ambiguity---------#
    data_sort=sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
    data_back=[];data_move=[]
    data_idx1 = [[item[0],item[1]] for item in data_sort]
    for i,k in zip(data_idx1,data_sort):
        if i not in data_back:
            data_back.append(i)
            data_move.append(k)
    nb_celllines = len(set([item[0] for item in data_move]))
    nb_drugs = len(set([item[1] for item in data_move]))
    print('Total %d pairs across %d cell lines and %d drugs.'%(len(data_move),nb_celllines,nb_drugs))

    return drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_move,nb_celllines,nb_drugs
