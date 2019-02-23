import argparse
from Tester import *

# pre_trained model path
pmp = '../model/ESC_50_ProtNet_att/BEST_MODEL'

# params for audio feature extraction (mel-spectrogram)
parser = argparse.ArgumentParser(description= 'PyTorch Implementation for few-shot sound recognition')
parser.add_argument('--dn',  default='ESC_50', type=str, help='dataset name')
parser.add_argument('--sr',  default=16000, type=int, help='[fea_ext] sample rate')
parser.add_argument('--ws',  default=2048,  type=int, help='[fea_ext] windows size')
parser.add_argument('--hs',  default=497,   type=int, help='[fea_ext] hop size')
parser.add_argument('--mel', default=128,   type=int, help='[fea_ext] mel bands')
parser.add_argument('--msc', default=5,     type=int, help='[fea_ext] top duration of audio clip')
parser.add_argument('--et',  default=10000, type=int, help='[fea_ext] spect manti')

# params for training
parser.add_argument('--bs',   default=5,    type=int,   help='[net] batch size')
parser.add_argument('--way',  default=5,    type=int,   help='[net] n-way')
parser.add_argument('--shot', default=5,    type=int,   help='[net] m-shot')
parser.add_argument('--lrde', default=20,    type=int,   help='[net] divided the learning rate 10 by every lrde epochs')
parser.add_argument('--mom',  default=0.9,   type=float, help='[net] momentum')
parser.add_argument('--wd',   default=1e-3,  type=float, help='[net] weight decay')
parser.add_argument('--lr',   default=0.01,   type=float, help='[net] learning rate')
parser.add_argument('--ep',   default=60*1,   type=int,   help='[net] epoch')
parser.add_argument('--beta', default=0.3,   type=float, help='[net] hyperparameter for pre-class loss weight')
parser.add_argument('--pmp',  default=pmp,   type=str,   help='[net] pre-trained model path')
args = parser.parse_args()

print args
print int((args.sr*args.msc)/args.hs)

Trer = Tester(args)
pred = Trer.run()




