import torch
import time
import argparse
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  
import numpy as np
from utils import Accumulator
from sklearn.metrics import accuracy_score,roc_auc_score
from utils import multi_class_auc



parser = argparse.ArgumentParser(description='DR Resnet Training')
parser.add_argument('-b','--batch_size',default=20,type=int,metavar='N')
parser.add_argument('--lr',default=3e-4,type=float,metavar='learning rate')
parser.add_argument('-e','--num_epoch',default=100,type=int,metavar='num of epoches')
parser.add_argument('--gpu',default=None,type=int,required=True,metavar='gpu num')
parser.add_argument('--ylim',default=1,type=int)
#parser.add_argument('--fig_name',default='resnet_tem.png',type=str)
parser.add_argument('--log_name',default='tem_log',type=str)
parser.add_argument('--test_epoch',default=1,type=int)
parser.add_argument('--num_workers',default=4,type=int)

parser.add_argument('--num_class', default=2, type=int)
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--crossCBAM", action="store_true")
parser.add_argument("--lambda_value", default=0.25, type=float)
#parser.add_argument("--adam", action="store_true")
parser.add_argument("--choice", default="both", type=str)

parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#def train(net, train_iter, test_iter, num_epochs, lr, args, writer):
def train(net, train_iter, test_iter, epoch, lr, args, writer, optimizer):
    device=torch.device(f'cuda:{args.gpu}')
    criterion=nn.CrossEntropyLoss().to(device) 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    net.train()
    #animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc', 'test auc'])
    num_batches = len(train_iter)
    
    
    start_time=time.time()

    metric = Accumulator(3)

    train_l_ave=0
    train_acc_ave=0
    print(f' epoch {epoch} start ')
    if epoch % 20 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
    #writer.add_scalar("lr", lr, epoch)
    for i, (X, y) in enumerate(train_iter):
        data_time.update(time.time()-start_time)
        #print(i,X.shape)
        
        X = X.to(device)
        y=[item.to(device) for item in y]
        output = net(X)
        
        loss1 = criterion(output[0], y[0])
        loss2 = criterion(output[1], y[1])
        loss3 = criterion(output[2], y[0])
        loss4 = criterion(output[3], y[1])
        loss = (loss1 + loss2 + args.lambda_value *loss3 + args.lambda_value * loss4)

        #print('loss:',loss)
        losses.update(loss.item(), X.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - start_time)
        start_time = time.time()
    
    
    return losses.avg


def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()
    device=torch.device(f'cuda:{args.gpu}')
    all_target = []
    all_target_dme = []
    all_output = []
    all_output_dme = []
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):      
            X = X.to(device)
            y=[item.to(device) for item in y]
            output = model(X)
            torch.cuda.synchronize()
            output0 = output[0]
            output1 = output[1]
            output0 = torch.softmax(output0, dim=1)
            output1 = torch.softmax(output1, dim=1)

            all_target.append(y[0].cpu().data.numpy())
            all_output.append(output0.cpu().data.numpy())
            all_target_dme.append(y[1].cpu().data.numpy())
            all_output_dme.append(output1.cpu().data.numpy())


    all_target = [item for sublist in all_target for item in sublist]
    all_output = [item for sublist in all_output for item in sublist]
    all_target_dme = [item for sublist in all_target_dme for item in sublist]
    all_output_dme = [item for sublist in all_output_dme for item in sublist]
    # acc
    acc_dr = accuracy_score(all_target, np.argmax(all_output,axis=1))
    acc_dme = accuracy_score(all_target_dme, np.argmax(all_output_dme, axis=1))

    # joint acc
    joint_result = np.vstack((np.argmax(all_output, axis=1), np.argmax(all_output_dme, axis=1)))
    joint_target = np.vstack((all_target, all_target_dme))
    joint_acc = ((np.equal(joint_result, joint_target) == True).sum(axis=0) == 2).sum() / joint_result.shape[1]

    # auc
    auc_dr = multi_class_auc(all_target, all_output, num_c = 5)
    ''' print(torch.Tensor(all_target_dme).shape,torch.Tensor(all_output_dme).shape)
    print(all_target_dme,all_output_dme)'''
    auc_dme = multi_class_auc(all_target_dme, all_output_dme, num_c = 2)


    return acc_dr, acc_dme, joint_acc, auc_dr, auc_dme
def main():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    print("  the porgram start now!  ")       
    
    from resnet50 import resnet50,resnet18,resnet101
    model = resnet50(num_classes=args.num_class, multitask=args.multitask, crossCBAM=args.crossCBAM, choice=args.choice)
    
    
    
    model_dict = model.state_dict()
    pretrain_path='/home/chengyuhan/ML/learn_code/pretrain/resnet50-19c8e357.pth'
    #pretrain_path='/home/chengyuhan/ML/learn_code/pretrain/resnet101-5d3b4d8f.pth'
    #pretrain_path='/home/chengyuhan/ML/learn_code/pretrain/resnet18-5c106cde.pth'
    pretrained_dict = torch.load(pretrain_path)
    pretrained_dict.pop('fc.weight', None)
    pretrained_dict.pop('fc.bias', None)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    #fc_features = model.fc.in_features
    #model.fc = nn.Linear(fc_features ,5)
    
    #print(model.fc)

    torch.cuda.set_device(args.gpu)
    model = model.to(torch.device(f'cuda:{args.gpu}'))

    from dataLoader import load_data
    batch_size=args.batch_size
    messidor_data_path='/home/chengyuhan/ML/data/DR/Messidor/IMAGES'
    messidor_label_path='/home/chengyuhan/ML/data/DR/Messidor/messidor_data.csv'
    train_itr,test_itr=load_data(messidor_data_path,messidor_label_path,batch_size,args.num_workers)

    writer = SummaryWriter(f'/home/chengyuhan/githubs/4471_project/DR_DME_joint/logs/{args.log_name}')
    
    lr, num_epochs = args.lr, args.num_epoch
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    best_acc1=0
    best_aucdr=0
    best_accdr=0
    for epoch in range(num_epochs):
        is_best = False
        is_best_auc = False
        is_best_acc = False
        # train for one epoch
        loss_train = train(model, train_itr, test_itr, epoch, lr, args, writer, optimizer)
        writer.add_scalar('Train loss', loss_train, epoch)

        # evaluate on validation set
        if epoch % args.test_epoch == 0:
            acc_dr, acc_dme, joint_acc, auc_dr, auc_dme = validate(test_itr, model, args)
            writer.add_scalar("Val acc_dr", acc_dr, epoch)
            writer.add_scalar("Val acc_dme", acc_dme, epoch)
            writer.add_scalar("Val acc_joint", joint_acc, epoch)
            writer.add_scalar("Val auc_dr", auc_dr, epoch)
            writer.add_scalar("Val auc_dme", auc_dme, epoch)

            best_acc1 = max(joint_acc, best_acc1)

            best_aucdr = max(auc_dr, best_aucdr)
            best_accdr = max(acc_dr, best_accdr)
    print('best joint acc: ',best_acc1,'best dr auc',best_aucdr,'best dr acc',best_accdr)
    
    
    
'''    plt.ylim(0,args.ylim)
    plt.savefig(args.fig_name,dpi=300)
    plt.show()'''


if __name__=="__main__":
    main()