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

class Trainer:
    def fit(self):
        st = time.time()
        save_dict = {}
        best_va_acc = 0
        best_te_acc = []
        self.model.train()
        for e in xrange(1, self.args.ep+1):
            # set optimizer (SGD)
            lr = self.args.lr * ( 0.1 **( (e% (self.args.lrde * 3)  ) /self.args.lrde) )
            print '\n==> Training Epoch #%d lr=%4f Best va_acc:%f'%(e, lr, best_va_acc)
            
            self.optimizer = optim.SGD(self.model.parameters(),
                    lr=lr, momentum=self.args.mom, weight_decay=self.args.wd)
            
            # Training
            for batch_idx, (X,_) in enumerate(self.tr_DS):
                X, Y = Variable(X.cuda()), Variable(self.Y.cuda())
                X = X.view(-1, 1, 128, 160)
                self.optimizer.zero_grad()
                pred = self.model(X, self.Xavg, self.Xstd, n=5, m=5)
                loss = self.mm_loss(pred, Y)
                loss.backward()
                self.optimizer.step()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                        %(e, self.args.ep, batch_idx+1, len(self.tr_DS),
                            loss.item(), time.time() - st))
                sys.stdout.flush()
            sys.stdout.write('\r')

            # Test
            self.result, va_acc = self.Tester(self.va_DS, 'Validation')
            self.result = self.Tester(self.te_DS, 'Test')

            if best_va_acc <= va_acc:
                best_va_acc = va_acc
                best_te_acc = self.result
                self.Saver()
                
            for i in best_te_acc:
                print '<==Best==> ' + i
    
    def mm_loss(self, pred, Y):
        return F.cross_entropy(pred, Y)

    def __init__(self, args):
        self.args = args
        ESC_X, ESC_Y, trvate = load_data(args.dn)
        tridx, vaidx, teidx = trvate
        
        self.Y = torch.LongTensor([self.args.way - 1]).repeat(args.bs)
        
        # build model
        self.model = nn.DataParallel(net().cuda())
        
        # data builder
        # default: n-ways m-shots
        self.tr_DS = B_DS(ESC_X, ESC_Y, tridx, self.args)
        self.va_DS = B_DS(ESC_X, ESC_Y, vaidx, self.args, mode='Test')
        te_DS_n5m5 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=5, m=5, mode='Test')
        te_DS_n5m1 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=5, m=1, mode='Test')
        te_DS_n10m5 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=10, m=5, mode='Test')
        te_DS_n10m1 = B_DS(ESC_X, ESC_Y, teidx, self.args, n=10, m=1, mode='Test')
        self.te_DS = [te_DS_n5m1, te_DS_n5m5, te_DS_n10m1, te_DS_n10m5]
        self.evl_nm = [[5, 1], [5, 5], [10, 1], [10, 5]]

        # load avg and std for Z-score 
        Xavg = torch.tensor(ESC_X[tridx].mean(keepdims=1).astype('float32'))
        Xstd = torch.tensor(ESC_X[tridx].std(keepdims=1).astype('float32'))
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())

        self.show_dataset_model_params()
    
    def Tester(self, DS, vate):
        st = time.time()
        self.model.eval()
        te_print = []
        for i in xrange(4):
            total = 0
            correct = 0

            if vate != 'Test':
                tar_DS = DS
                n, m = [5, 5]
            else:
                tar_DS = DS[i]
                n, m = self.evl_nm[i]
            
            for batch_idx, (X, _) in enumerate(tar_DS):
                total += X.size(0)
                X, Y = Variable(X.cuda()), Variable(self.Y.cuda())
                X = X.view(-1, 1, 128, 160)
                pred = self.model(X, self.Xavg, self.Xstd, n, m)

                # Max
                _, pred = torch.max(pred.data, 1)
                correct += (pred == n - 1).sum().item()

            oprint = '%s %d-way %d-shot acc:%f Time:%1f'%(vate, n, m, correct/float(total), time.time() - st)
            print oprint
            te_print.append(oprint)
            if vate != 'Test':
                return oprint, correct / float(total)
        
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

            print 'Load Pre_trained Model : ' + self.args.pmp

        else:
            print 'Learning from scrath'

    def show_dataset_model_params(self):
        # show model structure
        print self.model
        # show params
        print show_model_params(self.model)

    def Saver(self):
        save_dict = {}
        directory = '../model/%s_%s'%(self.args.dn, self.model.module.model_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        save_dict['state_dict'] = self.model.state_dict()
        save_dict['result'] = self.result
        torch.save(save_dict, directory + '/BEST_MODEL')
        print 'Save the best model to %s' % (directory)

