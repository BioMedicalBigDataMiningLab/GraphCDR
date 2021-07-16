#Extract Drug features through Deepchem
import os
import rdkit
import deepchem as dc
from rdkit import Chem
import hickle as hkl
'''
CanonicalSMILES = 'CC1CCCC2(C(O2)CC(OC(=O)CC(C(C(=O)C(C1O)C)(C)C)O)C(=CC3=CSC(=N3)C)C)C'
mol = Chem.MolFromSmiles(CanonicalSMILES)
Simles=Chem.MolToSmiles(mol)
'''
drug_smiles_file='../../data/Drug/222drugs_pubchem_smiles.txt'
save_dir='drug_graph_feat'
pubchemid2smile = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open(drug_smiles_file).readlines()}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
molecules = []
for each in pubchemid2smile.keys():
	molecules=[]
	molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
	featurizer = dc.feat.graph_features.ConvMolFeaturizer()
	mol_object = featurizer.featurize(mols=molecules)
	features = mol_object[0].atom_features
	degree_list = mol_object[0].deg_list
	adj_list = mol_object[0].canon_adj_list
	hkl.dump([features,adj_list,degree_list],'%s/%s.hkl'%(save_dir,each))





