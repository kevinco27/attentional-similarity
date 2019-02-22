import numpy as np
import SharedArray as sa
import sys
from collections import OrderedDict

def f_load(m_name, fp):
    try:
        out = sa.attach(m_name)
    except:
        out = np.load(fp)
        X = sa.create(m_name, (out.shape), dtype='float32')
        X[:] = out
    return out.astype('float32')

def load_data(data_name):
    if 'noise' not in data_name:
        fp = '../data/ESC_sep.npy'
        print 'Load %s from %s' % (data_name, fp)
        ESC_all_X = f_load('ESC_all_sep_X', fp)
    else:
        fp = '../data/ESC_noise_sep.npy'
        print 'Load %s from %s' % (data_name, fp)
        ESC_all_X = f_load('ESC_noise_sep_X' , fp)

    fp = '../data/ESC_tag.npy'
    ESC_all_Y = f_load('ESC_all_Y' , fp)

    tag2idx = np.load('../data/ESC_tag2idx.npy').item()
    tag2idx = OrderedDict(sorted(tag2idx.iteritems(), key=lambda x: x[1]))
    for idx, (k, v) in enumerate(tag2idx.iteritems()):
        print '%s:%s' % (k, v)
    
    rand_tag = np.arange(50)
    RA = np.random.RandomState(0)
    RA.shuffle(rand_tag)
    
    # 35 class for training, 5 for validating and 10 for testing
    tr_tag = rand_tag[:35]
    va_tag = rand_tag[35: 35 + 5]
    te_tag = rand_tag[40:]
    
    print 'Classes for training: %s' % tr_tag
    print 'Classes for testing: %s' % te_tag
    print 'Classes for validating: %s' % va_tag
    idx = np.arange(40)
    tr_idx = np.tile(idx, len(tr_tag)) + np.repeat(tr_tag*40, 40)
    va_idx = np.tile(idx, len(va_tag)) + np.repeat(va_tag*40, 40)
    te_idx = np.tile(idx, len(te_tag)) + np.repeat(te_tag*40, 40)

    return ESC_all_X, ESC_all_Y, [tr_idx, va_idx, te_idx]
        
        

