import numpy as np
import torch
from torch import nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
      
      
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    """_summary_

    Returns:
        _type_: _description_
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype)==y
    tem01=cmp.type(y.dtype)
    sum_total=float(tem01.sum())
    return sum_total
def evaluate(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            #print(net(X))
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

  


def multi_class_auc(all_target, all_output, num_c = None):
    all_output = np.stack(all_output)
    if num_c>2:
        all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    else:
        numm=len(all_target)
        all_target_binarize=[]
        for i in range(numm):
            if all_target[i]==0:
                all_target_binarize.append([1,0])
            else:
                all_target_binarize.append([0,1])
        all_target=np.array(all_target_binarize)
    #print(torch.Tensor(all_target).shape,torch.Tensor(all_output).shape)
    auc_sum = []
    for num_class in range(num_c):
        try:
            #print(all_target[:, num_class], all_output[:, num_class])
            auc = metrics.roc_auc_score(all_target[:, num_class], all_output[:, num_class])
            auc_sum.append(auc)
        except ValueError:
            pass
    auc = sum(auc_sum) / float(len(auc_sum))
        #print(torch.Tensor(all_target).shape,torch.Tensor(all_output_tar).shape)
    return auc
def auc_softmax(y_hat,y,device,class_num):
  #y_hat=y_hat.to('cpu')
  mm=nn.Softmax(dim=1)
  mm.to(device)
  y_score=mm(y_hat)
  y_score=y_score.to('cpu').numpy()
  y=y.to('cpu').numpy()
  '''y_prob=[]
  y_prob=y_score[:,1]
  y_prob=np.array(y_prob)'''
  #print('auc:     ',metrics.roc_auc_score(y,y_prob))
  return multi_class_auc(y,y_score,class_num)
  #return metrics.roc_auc_score(y,y_prob)
def eva_auc(net, data_iter, device=None, class_num=5):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            #print(net(X))
            print('auc',auc_softmax(net(X), y, device, class_num))
            metric.add(auc_softmax(net(X), y, device, class_num),1)
    print('met',metric[0]/metric[1])
    return metric[0]/metric[1]