import os
import sys
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.optim as optim
from scipy import io as scio
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import DGCNN
# from tqdm import tqdm
from utils import eegDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './' #setting the environment variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def load_eegdata(file_path):
    raw = scio.loadmat(file_path)  #生成字典
    DE_theta = raw['DE_theta']
    DE_alpha = raw['DE_alpha']
    DE_beta = raw['DE_beta']
    DE_gamma = raw['DE_gamma']
    label = raw['label']

    index1 = np.where(label == 1)[0]   #条件满足 返回满足条件的坐标  （以元组的形式，里面是数组）   https://www.jb51.net/article/260293.htm  原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。例如二维，一一组合对应坐标几行几列
    index2 = np.where(label == 2)[0]
    index3 = np.where(label == 3)[0]
###去掉0标签对应的数据
    DE_theta = np.concatenate([DE_theta[index1, :], DE_theta[index2, :], DE_theta[index3, :]], axis=0)
    DE_alpha = np.concatenate([DE_alpha[index1, :], DE_alpha[index2, :], DE_alpha[index3, :]], axis=0)
    DE_beta = np.concatenate([DE_beta[index1, :], DE_beta[index2, :], DE_beta[index3, :]], axis=0)
    DE_gamma = np.concatenate([DE_gamma[index1, :], DE_gamma[index2, :], DE_gamma[index3, :]], axis=0)
    label = np.concatenate([np.zeros([len(index1)]), np.zeros([len(index2)])+1, np.zeros([len(index3)])+2], axis=0)

    ###x = np.zeros([len(index1)])   一维数组是“列向量”
    DE_theta = np.expand_dims(DE_theta, axis=2)
    DE_alpha = np.expand_dims(DE_alpha, axis=2)
    DE_beta = np.expand_dims(DE_beta, axis=2)
    DE_gamma = np.expand_dims(DE_gamma, axis=2)




##########打乱
    index = np.random.permutation(DE_theta.shape[0])   #随机排列一个序列，或者数组。
                                                        #如果x是多维数组，则沿其第一个坐标轴的索引随机排列数组。
    DE_theta = DE_theta[index, :, :]
    DE_alpha = DE_alpha[index, :, :]
    DE_beta = DE_beta[index, :, :]
    DE_gamma = DE_gamma[index, :, :]
    label = label[index]

    data = np.concatenate([DE_theta, DE_alpha, DE_beta, DE_gamma], axis=2)

    return data, label


def load_dataloader(data_train, data_test, label_train, label_test):
    batch_size = 64
    train_iter = DataLoader(dataset=eegDataset(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    test_iter = DataLoader(dataset=eegDataset(data_test, label_test),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1)

    return train_iter, test_iter


def train(train_iter, test_iter, model, criterion, optimizer, num_epochs):
    # Train
    
    print('began training on', device, '...')

    acc_test_best = 0.0
    n = 0
    for ep in range(num_epochs):
        model.train()
        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_iter:

            images = images.float().to(device)
            labels = labels.to(device)

            # Compute loss & accuracy
            output = model(images)


            loss = criterion( output, labels.long())          #标签从0开始！！！！！！！！！！！！！！必须要保证label是0开始的连续整数，因为label是一种索引
            

            #correct = 0
            #a = len(labels)
            pred = output.argmax(dim=1)
            
            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            #scheduler.step()
            optimizer.step()
            #print(optimizer.state_dict)
            optimizer.zero_grad()
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                    batch_id,
                                                                    total_loss / batch_id,
                                                                    accuracy))
            batch_id += 1

        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

        acc_test = evaluate(test_iter, model)

        if acc_test >= acc_test_best:
            n = 0
            acc_test_best = acc_test
            model_best = model

        # 学习率逐渐下降，容易进入局部最优，当连续10个epoch没有跳出，且有所下降，强制跳出
        # if n >=  num_epochs//10 and acc_test < 0.99: 
        if n >=  num_epochs//10 and acc_test < acc_test_best-0.1: 
            print('#########################reload#########################')
            n = 0
            model = model_best
        # find best test acc model in all epoch(not last epoch)

        print('>>> best test Accuracy: {}'.format(acc_test_best))

    return acc_test_best


def evaluate(test_iter, model):
    # Eval
    print('began test on', device, '...')
    model.eval()
    correct, total = 0, 0
    for images, labels in test_iter:
        # Add channels = 1
        images = images.float().to(device)
    
        # Categogrical encoding
        labels = labels.to(device)
        
        output = model(images)
        
        pred = output.argmax(dim=1)


        correct += (pred == labels).sum().item()
        total += len(labels)
    print('test Accuracy: {}'.format(correct / total))
    return correct / total


def main():
    dir = r'./data/S_21.mat'                         #04-0.9916， 0.86    
    data, label = load_eegdata(dir)    #data （‘采样点’，通道，4频带， 1080 59 4）   lable  对应‘采样点’的标签 1080

    xdim = [64, 59, 4]
    k_adj = 40
    num_out = 64


    num_epochs = 10
    fold_num = 10
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=2)   
    kf_i = 0
    acc_mean = 0
    for train_index, test_index in kf.split(data):     #864 ： 216
        
        if device.type == 'cuda':
            print('empty cuda cache...')
            torch.cuda.empty_cache()

        model = DGCNN(xdim, k_adj, num_out).to(device)
   
        criterion = nn.CrossEntropyLoss().to(device) #使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
        optimizer = optim.Adam(model.parameters(),
                            lr=0.001,
                            weight_decay=0.0001)
                            #momentum=0.9)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)  
        kf_i = kf_i + 1
        print("第", kf_i, "折")

        ## 训练的数据要求导，必须使用torch.tensor包装
        data_train = torch.tensor(data[train_index, :, :])
        label_train = torch.tensor(label[train_index])


        data_test = data[test_index, :, :]
        label_test = label[test_index]


        train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test)
        acc_test_best = train(train_iter, test_iter, model, criterion, optimizer, num_epochs)
        acc_mean = acc_mean + acc_test_best
        
    acc_mean = acc_mean / fold_num
    print('>>> ' + str(fold_num) + ' fold test acc: ', acc_mean)


if __name__ == '__main__':
    sys.exit(main())