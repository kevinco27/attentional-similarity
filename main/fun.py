import torch
import torch.nn.init as init

def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, mode='fan_out')
        #init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, mode='fan_out')
        init.constant_(m.bias, 0)

def show_model_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print 'Model:' + model.module.model_name + '\t#params:%d'%(params)



