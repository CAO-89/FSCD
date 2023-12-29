# https://github.com/tuantle/regression-losses-pytorch
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import cohen_kappa_score

##############2D Losses
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
    
        #flatten label and prediction tensors
        inputs = inputs.argmax(dim=1).view(-1).float()
        targets = targets.view(-1).float()
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
        
class DiceCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()
        
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs_fl = inputs.argmax(dim=1).view(-1).float()
        targets_fl = targets.view(-1).float()
        
        intersection = (inputs_fl * targets_fl).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_fl.sum() + targets_fl.sum() + smooth)  
        CE = F.cross_entropy(inputs, targets, self.weight, reduction='mean')

        return CE + dice_loss
        
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, alpha=0.5, gamma=2, smooth=1):
        
        inputs = inputs.float() 
        
        #first compute cross-entropy 
        ce_loss = F.cross_entropy(inputs, targets,reduction='mean',weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        
        return focal_loss

class IoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.argmax(dim=1).view(-1).float()
        targets = targets.view(-1).float()
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def choose_criterion2d(name, class_weights, class_ignored = -99999):
    if name == 'bce':
        return torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index=class_ignored)
    elif name == 'dice':
        return DiceLoss(weight = class_weights)
    elif name == 'dicece':
        return DiceCELoss(weight = class_weights)
    elif name == 'jaccard':
        return IoULoss(weight = class_weights)
    elif name == 'focal':
        return FocalLoss(weight = class_weights)

def calMetric_iou(predict, label):


    tp = torch.sum(torch.logical_and(predict == 1, label == 1))
    fp = torch.sum(torch.logical_and(predict == 1,label != 1))
    tn = torch.sum(torch.logical_and(predict != 1,label != 1))
    fn = torch.sum(torch.logical_and(predict != 1,label == 1))


    return tp,fp,tn,fn
