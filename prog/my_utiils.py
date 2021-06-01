import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)
def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)
def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1-acc
    tp = (yt * yp).sum()
    tn = ((1 - yt) * (1 - yp)).sum()
    fp = ((1 - yt) * yp).sum()
    fn = (yt * (1 - yp)).sum()
    epsilon = 0.0000001
    accuracy= (tp + tn) / (tp + tn + fn + fp)
    precision2 = tp / (tp + fp + epsilon)
    recall2 = tp / (tp + fn + epsilon)
    f1 = 2 * (precision2 * recall2) / (precision2 + recall2 + epsilon)
    return auc,aupr,f1,accuracy,precision2,recall2
