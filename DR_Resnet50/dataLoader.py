import os
import pandas as pd
import random
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms
from sklearn.preprocessing import label_binarize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
size  = 224
tra = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            #transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            normalize,
        ])
class dataSet(data.Dataset):
  def __init__(self, data_tensor, label_tensor):
    self.data_tensor = data_tensor
    self.label_tensor = label_tensor

  def __getitem__(self, index):
    img=self.data_tensor[index]
    label=self.label_tensor[index]
    img=tra(img)
    return img,label

  def __len__(self):
        return len(self.data_tensor)

def ex_image_id(s):
    return s[:s.index('.')]
def get_label_DR(sheet,img_name):
    for i in range(len(sheet)):
        if ex_image_id(img_name)==ex_image_id(sheet.loc[i][0]):
            #print(sheet.loc[i][1])
            if int(sheet.loc[i][3])==0:
                return -1
            label=int(sheet.loc[i][1])
            return label
def get_label_DME(sheet,img_name):
    for i in range(len(sheet)):
        if ex_image_id(img_name)==ex_image_id(sheet.loc[i][0]):
            #print(sheet.loc[i][1])
            if int(sheet.loc[i][3])==0:
                return -1
            label=int(sheet.loc[i][2])
            return label
def load_data(path_img,path_label,batch_size,num,task):
    data_tensor=[]
    label_tensor=[]
    images=os.listdir(path_img)
    sheet=pd.read_csv(path_label)
    #print(len(sheet))
    #print(get_label(sheet,'IM003718.jpg'))
    for image_name in images:
        img=Image.open(f'{path_img}/{image_name}')#////////////////////////
        img = img.convert('RGB')
        #print(image_name)
        if task=='DR':
            label=get_label_DR(sheet,image_name)
        elif task=='DME':
            label=get_label_DME(sheet,image_name)
        if label==-1:
            continue
        #print(image_name,label)
        data_tensor.append(img)
        label_tensor.append(label)
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(data_tensor)
    random.seed(randnum)
    random.shuffle(label_tensor)
    train_data=data_tensor[100:]
    train_label=label_tensor[100:]
    test_data=data_tensor[:100]
    test_label=label_tensor[:100]
    train_tensor_dataset=dataSet(train_data,train_label)
    test_tensor_dataset=dataSet(test_data,test_label)
    print("train: ",train_tensor_dataset.__len__(),"  test: ",test_tensor_dataset.__len__())
    return data.DataLoader(train_tensor_dataset,batch_size,shuffle=True,num_workers=num),data.DataLoader(test_tensor_dataset,batch_size,shuffle=True,num_workers=num)
    #print(data_tensor,label_tensor)