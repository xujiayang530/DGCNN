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
from scipy.stats import zscore
from model import DGCNN
# from tqdm import tqdm
from utils import eegDataset

import os
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './' #setting the environment variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_DE_SEED(load_path):
    filePath = load_path 
    datasets = scio.loadmat(filePath)
    DE = datasets['DE']
    # DE_delta = np.squeeze(DE[:,:,0]).T
    # DE_theta = np.squeeze(DE[:,:,1]).T
    # DE_alpha = np.squeeze(DE[:,:,2]).T
    # DE_beta = np.squeeze(DE[:,:,3]).T
    # DE_gamma = np.squeeze(DE[:,:,4]).T
    # dataAll = np.concatenate([DE_delta,DE_theta,DE_alpha,DE_beta,DE_gamma], axis=1)
    dataAll = np.transpose(DE, [1,0,2])
    labelAll = datasets['labelAll'].flatten()

    labelAll = labelAll + 1

    return dataAll, labelAll


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


def train(train_iter, test_iter, model, criterion, optimizer, num_epochs, sub_name):
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



def main_LOCV():
    dir = './SEED/SEED_code/DE/session1/'                         #04-0.9916， 0.86    
    # os.chdir(dir) # 可能在寻找子文件的时候路径进了data
    file_list = os.listdir(dir)
    sub_num = len(file_list)

    xdim = [128, 62, 5] # batch_size * channel_num * freq_num
    k_adj = 40
    num_out = 64
    num_epochs = 20
    acc_mean = 0
    acc_all = []
    for sub_i in range(sub_num):

        
        load_path = dir + file_list[sub_i]  # ../表示上一级目录
        data_test, label_test = load_DE_SEED(load_path)    #data （‘采样点’，通道，4频带， 1080 59 4）   lable  对应‘采样点’的标签 1080

        # if device.type == 'cuda':
        #         print('empty cuda cache...')
        #         torch.cuda.empty_cache()

        data_test = zscore(data_test)

        model = DGCNN(xdim, k_adj, num_out).to(device)


        criterion = nn.CrossEntropyLoss().to(device) #使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
        optimizer = optim.Adam(model.parameters(),
                            lr=0.001,
                            weight_decay=0.0001)
                            #momentum=0.9)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        train_list = copy.deepcopy(file_list)
        train_list.remove(file_list[sub_i])  
        train_num = len(train_list)

        data_train = []
        label_train = []
        
        for train_i in range(train_num):     


            train_path = dir + train_list[train_i]
            data, label = load_DE_SEED(train_path)

            data = zscore(data)

            if train_i == 0:
                data_train = data
                label_train = label
            else:
                data_train = np.concatenate((data_train, data), axis=0)
                label_train = np.concatenate((label_train, label), axis=0)
        

        # data_train = zscore(data_train)


        ## 训练的数据要求导，必须使用torch.tensor包装
        data_train = torch.tensor(data_train)
        label_train = torch.tensor(label_train)

        train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test)
        acc_test_best = train(train_iter, test_iter, model, criterion, optimizer, num_epochs, file_list[sub_i])
        acc_mean = acc_mean + acc_test_best/sub_num
        acc_all.append(acc_test_best)


    print('save...')
    scio.savemat('./result/acc_all/acc_de_SEED_LOCV.mat',{'acc_all':acc_all,\
    'sub_list':np.array(file_list,dtype=np.object)})


    print('>>> LOSV test acc: ', acc_all)
    print('>>> LOSV test mean acc: ', acc_mean)
    print('>>> LOSV test std acc: ', np.std(np.array(acc_all)))


if __name__ == '__main__':
    sys.exit(main_LOCV())
