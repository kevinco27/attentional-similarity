import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import time
import sys
sys.path.append('./net')
from fun import *
from protnet_att import *
from loader import *
from data_gentor import *

class Tester:
    def run(self):
        self.Tester()
        return self.Show_pred()
    
    def Show_embed(self):
        self.model.eval()
        pred = []
        for i in xrange(1):
            total = 0
            tar_DS = self.te_DS[i]
            n,m = self.evl_nm[i]
            out = np.zeros((400, 32*4*3))
            out1 = np.zeros((400, 6))
            for batch_idx, (X, s_idx) in enumerate(tar_DS):
                s_idx = s_idx.numpy()
                out1[total:total+2,:6] = s_idx
                
                X, Y = Variable(X.cuda()), Variable(self.Y.cuda())
                X = X.view(-1, 1, 128,160)
                pred  = self.model(X, self.Xavg, self.Xstd, n, m)
                
                #_, pred = torch.max(pred.data, 1)
                
                out[total:total+2,:] = pred.data
                total += 2
        return out, out1
    
    def Show_pred(self):
        self.model.eval()
        pred = []
        for i in xrange(1):
            total = 0
            tar_DS = self.te_DS[i]
            n,m = self.evl_nm[i]
            out = np.zeros((400, 7))
            for batch_idx, (X, s_idx) in enumerate(tar_DS):
                s_idx = s_idx.numpy()
                out[total:total+2,:6] = s_idx
                
                X, Y = Variable(X.cuda()), Variable(self.Y.cuda())
                X = X.view(-1, 1, 128,160)
                pred  = self.model(X, self.Xavg, self.Xstd, n, m)
                
                _, pred = torch.max(pred.data, 1)
                
                out[total:total+2,-1] = pred.data
                total += 2
        return out

    def __init__(self, args):
        self.args = args
        # load data tidx=[tr_idx, tr_nidx, te_idx, te_nidx]
        ESC_X, ESC_Y, trvate = load_data(args.dn)
        tridx, vaidx, teidx = trvate
        
        self.Y = torch.LongTensor([self.args.way -1]).repeat(args.bs)
        self.model = nn.DataParallel(net().cuda())
        
        # data builder
        # default: n-ways m-shots
        self.tr_DS = B_DS(ESC_X, ESC_Y, tridx, self.args)
        self.va_DS = B_DS(ESC_X, ESC_Y, vaidx, self.args, mode='Test')
        te_DS_n5m5  = B_DS(ESC_X, ESC_Y, teidx, self.args, n= 5, m=5, mode='Test')
        te_DS_n5m1  = B_DS(ESC_X, ESC_Y, teidx, self.args, n= 5, m=1, mode='Test')
        te_DS_n10m5 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=10, m=5, mode='Test')
        te_DS_n10m1 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=10, m=1, mode='Test')
        self.te_DS = [te_DS_n5m1, te_DS_n5m5, te_DS_n10m1, te_DS_n10m5]
        self.evl_nm = [[5,1], [5,5], [10,1], [10,5]]

        # load avg and std for Z-score 
        Xavg = torch.tensor(ESC_X[tridx].mean(keepdims=1).astype('float32'))
        Xstd = torch.tensor(ESC_X[tridx].std(keepdims=1).astype('float32'))
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())

        self.show_dataset_model_params()
        self.load_pretrained_model(self.model)
    
    def Tester(self , vate='Test'):
        print '\n'
        st = time.time()
        self.model.eval()
        te_print = []
        for i in xrange(4):
            total = 0
            correct = 0
            tar_DS = self.te_DS[i]
            n,m = self.evl_nm[i]
            
            for batch_idx, (X, _) in enumerate(tar_DS):
                total += X.size(0)
                X, Y = Variable(X.cuda()), Variable(self.Y.cuda())
                X = X.view(-1, 1, 128,160)
                pred  = self.model(X, self.Xavg, self.Xstd, n, m)

                # Max
                _, pred = torch.max(pred.data, 1)
                correct += (pred==n-1).sum().item()
            oprint = '%s %d-way %d-shot acc:%f Time:%1f'%(vate, n, m, correct/float(total), time.time() - st)
            print oprint
            te_print.append(oprint)
            if vate != 'Test':
                return oprint, correct/float(total)
        
        return te_print
    

    def load_pretrained_model(self, model):
        # pre-training
        if os.path.exists(self.args.pmp):
            pretrained_model = torch.load(self.args.pmp)
            model_param = model.state_dict()
            for k in pretrained_model['state_dict'].keys():
               try:
                    model_param[k].copy_(pretrained_model['state_dict'][k])
                    print k
               except:
                    print '[ERROR] Load pre-trained model %s'%(k)
                    #self.model.apply(model_init)
                    #break
            print 'Load Pre_trained Model : ' + self.args.pmp
        
        else:
            print 'Learning from scrath'
            #self.model.apply(model_init)
            

    def show_dataset_model_params(self):
        # show model structure
        print self.model
        # show params
        print show_model_params(self.model)

