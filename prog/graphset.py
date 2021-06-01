from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features, edge_index = data_mol[0],data_mol[1]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA