import torch
import time
import argparse
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  

from utils import Accumulator
from utils import evaluate
from utils import eva_auc
from utils import accuracy


parser = argparse.ArgumentParser(description='DR Resnet Training')
parser.add_argument('-b','--batch_size',default=20,type=int,metavar='N')
parser.add_argument('--lr',default=3e-4,type=float,metavar='learning rate')
parser.add_argument('-e','--num_epoch',default=100,type=int,metavar='num of epoches')
parser.add_argument('--gpu',default=None,type=int,required=True,metavar='gpu num')
parser.add_argument('--ylim',default=1,type=int)
#parser.add_argument('--fig_name',default='resnet_tem.png',type=str)
parser.add_argument('--log_name',default='tem_log',type=str)
parser.add_argument('--test_epoch',default=10,type=int)
parser.add_argument('--num_workers',default=2,type=int)
parser.add_argument('--task',default='DR',type=str)

def train(net, train_iter, test_iter, num_epochs, lr, args, writer):
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    #animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc', 'test auc'])
    num_batches = len(train_iter)
    device=torch.device(f'cuda:{args.gpu}')
    for epoch in range(num_epochs):
        tem_time=time.time()
        # Sum of training loss, sum of training accuracy, no. of examples
        print(epoch)
        metric = Accumulator(3)
        net.train()
        train_l_ave=0
        train_acc_ave=0
        print(' epoch start ')
        if epoch % 30 == 0:
          for p in optimizer.param_groups:
            p['lr'] *= 0.1
        for i, (X, y) in enumerate(train_iter):
            #print(i,X.shape)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            train_l_ave+=train_l
            train_acc_ave+=train_acc
            '''if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))'''
        writer.add_scalar('train_loss', train_l_ave/num_batches, epoch)
        writer.add_scalar('train_acc', train_acc_ave/num_batches, epoch)
        if epoch % args.test_epoch == 0 or epoch==num_epochs-1:
            test_acc = evaluate(net, test_iter)
            if args.task=='DR':
                class_num=5
            elif args.task=='DME':
                class_num=2
            test_auc = eva_auc(net,test_iter,device,class_num)
            print("acc: ",test_acc,"  auc: ",test_auc)
            #animator.add(epoch + 1, (train_l_ave/num_batches, train_acc_ave/num_batches, test_acc, test_auc))
            writer.add_scalar('test_acc',test_acc,epoch)
            writer.add_scalar('test_auc',test_auc,epoch)
        #else:
            #animator.add(epoch + 1, (train_l_ave/num_batches, train_acc_ave/num_batches, None, None))
        print(f'eopch cost:{time.time() - tem_time:.4f}s')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}, test_auc {test_auc:.3f}')
def main():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    print("  the porgram start now!  ")       
    
    model=torchvision.models.resnet50(pretrained=True)
    model_dict = model.state_dict()
    pretrain_path='/home/chengyuhan/ML/learn_code/pretrain/resnet50-19c8e357.pth'
    pretrained_dict = torch.load(pretrain_path)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features ,5)
    #print(model.fc)

    torch.cuda.set_device(args.gpu)
    model = model.to(torch.device(f'cuda:{args.gpu}'))

    from dataLoader import load_data
    batch_size=args.batch_size
    messidor_data_path='/home/chengyuhan/ML/data/DR/Messidor/IMAGES'
    messidor_label_path='/home/chengyuhan/ML/data/DR/Messidor/messidor_data.csv'
    train_itr,test_itr=load_data(messidor_data_path,messidor_label_path,batch_size,args.num_workers,args.task)

    writer = SummaryWriter(f'/home/chengyuhan/githubs/4471_project/DR_Resnet50/logs/{args.log_name}')
    
    lr, num_epochs = args.lr, args.num_epoch
    train(model, train_itr, test_itr, num_epochs, lr, args, writer)
'''    plt.ylim(0,args.ylim)
    plt.savefig(args.fig_name,dpi=300)
    plt.show()'''


if __name__=="__main__":
    main()