import torch
from torch.utils.data import Dataset
from extractor import *

# data loader

class tr_data(Dataset):
    def __init__(self, X,Y, didx, args, n, m, mode):
        self.X = X
        self.Y = Y
        self.didx = didx
        self.args = args
        self.get_tag_idx = np.unique(Y[didx])
        self.mode = mode
        self.n = n
        self.m = m
    
    def choice(self, sup, sel):
        if self.mode != 'Training':
            RA = np.random.RandomState(self.dindex)
            Support_tag = RA.choice(sup, sel, replace=False)
        else:
            Support_tag = np.random.choice(sup, sel, replace=False)
        
        return Support_tag

    def __getitem__(self, index):
        dindex = self.didx[index]
        self.dindex = dindex
        
        # Query 
        Query_tag = self.Y[dindex]
        
        # 5-way (5 classes)
        #way = self.args.way
        way = self.n
        
        Support_tag = self.choice(self.get_tag_idx, way)
        Support_tag = np.setdiff1d(Support_tag, Query_tag)[:way-1]
        
        # 5-shot (5 smaples per class)
        #shot = self.args.shot
        shot = self.m
        
        Support_set = []
        for tag in Support_tag:
            # only for ESC
            Support_set.extend(self.choice(40, shot) + tag*40)
        
        Query_set = self.choice(40, shot+1) + Query_tag*40
        Query_set = np.setdiff1d(Query_set, dindex)[:shot]
        
        Support_set.extend(Query_set)
        Support_set.extend([dindex])
        Support_set = np.array(Support_set).astype(int)   
        
        if self.mode == 'Training':
            rate = np.random.randint(20, 160 - 20)
            return np.roll(self.X[Support_set], rate, axis=-1), Support_set
        else:
            return self.X[Support_set], Support_set
    
    def __len__(self):
        return len(self.didx)

def B_DS(X, Y, didx, args, n=5, m=5, mode='Training'):
    #  n:n-ways m: m-shots
    if mode == 'Training':
        kwargs = {'batch_size': args.bs, 'num_workers': 30, 'pin_memory': True,
        'drop_last': True}
        tr_DS = tr_data(X, Y, didx, args, n, m, mode)
        tr_loader = torch.utils.data.DataLoader(tr_DS, shuffle=True, **kwargs)
        return tr_loader
    
    else:
        kwargs = {'batch_size': 2, 'num_workers': 30, 'pin_memory': True}
        te_DS = tr_data(X, Y, didx, args, n, m, mode)
        te_loader = torch.utils.data.DataLoader(te_DS, **kwargs)
        return te_loader



