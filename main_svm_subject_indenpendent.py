import scipy.io as scio
import numpy as np
import random
from sklearn import feature_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import os
import copy


def classcify(data_train, label_train, data_test, label_test):


    model = SVC(kernel='linear')
    model.fit(data_train, label_train)
    label_pre = model.predict(data_test)
    acc_end = sum(label_test == label_pre) / len(label_test)

    f1_end = f1_score(label_test, label_pre, average='weighted')
    return acc_end, f1_end



def balance(data, label):

    flag = 1
    index1 = np.where(label == 1)[0]
    index2 = np.where(label == 2)[0]
    index3 = np.where(label == 3)[0]

    index_min = np.min([len(index1), len(index2), len(index3)])

    if index_min == 0:
        flag = 0
        return flag, data, label

    data1 = data[index1[0:index_min], :]
    data2 = data[index2[0:index_min], :]
    data3 = data[index3[0:index_min], :]

    label1 = label[index1[0:index_min]]
    label2 = label[index2[0:index_min]]
    label3 = label[index3[0:index_min]]

    dataAll = np.concatenate([data1, data2, data3], axis=0)
    labelAll = np.concatenate([label1, label2, label3], axis=0)

    # 打乱
    index = np.random.permutation(dataAll.shape[0])
    dataAll = dataAll[index, :]
    labelAll = labelAll[index]

    return flag, dataAll, labelAll


def zscore(dataAll):

    data_new = np.zeros([dataAll.shape[0], dataAll.shape[1]])

    for trial_i in range(dataAll.shape[0]):
        xx = np.squeeze(dataAll[trial_i, :])
        data_new[trial_i, :] = (xx - np.mean(xx)) / np.var(xx)


    return data_new

def load_eegdata(filePath):
    datasets = scio.loadmat(filePath)
    DE_theta = datasets['DE_theta']
    DE_alpha = datasets['DE_alpha']
    DE_beta = datasets['DE_beta']
    DE_gamma = datasets['DE_gamma']
    label = datasets['label'].flatten()


    index1 = np.where(label==1)[0]
    index2 = np.where(label==2)[0]
    index3 = np.where(label==3)[0]

    DE_theta = np.concatenate([DE_theta[index1,:],DE_theta[index2,:],DE_theta[index3,:]], axis=0)
    DE_alpha = np.concatenate([DE_alpha[index1,:],DE_alpha[index2,:],DE_alpha[index3,:]], axis=0)    
    DE_beta = np.concatenate([DE_beta[index1,:],DE_beta[index2,:],DE_beta[index3,:]], axis=0)
    DE_gamma = np.concatenate([DE_gamma[index1,:],DE_gamma[index2,:],DE_gamma[index3,:]], axis=0)
    label = np.concatenate([np.zeros(len(index1)),np.zeros(len(index2))+1,np.zeros(len(index3))+2], axis=0)

    dataAll = np.concatenate([DE_theta,DE_alpha,DE_beta,DE_gamma], axis=1)

    index = np.random.permutation(dataAll.shape[0])
    dataAll = dataAll[index, :]
    label = label[index]

    return dataAll, label

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
    
    dataAll = np.reshape(dataAll,[dataAll.shape[0],-1])

    labelAll = labelAll + 1

    return dataAll, labelAll


load_path = "./SEED/SEED_code/DE/session1/"

os.chdir(load_path)
file_list = os.listdir()
sub_num = len(file_list)

acc_all = []
f1_all = []

for sub_i in range(sub_num):

    print('subject test>>>', file_list[sub_i])
    # data_test, label_test = load_eegdata('.' + load_path + file_list[sub_i])
    data_test, label_test = load_DE_SEED(load_path + file_list[sub_i])

    train_list = copy.deepcopy(file_list)
    train_list.remove(file_list[sub_i])
    train_num = len(train_list)

    data_train = []
    label_train = []

    for train_i in range(train_num):

        print('subject train', train_list[train_i])
        data_temp, label_temp = load_de1122(load_path + train_list[train_i])

        if train_i == 0:
            data_train = data_temp
            label_train = label_temp
        else:
            data_train = np.concatenate([data_train, data_temp], axis=0)
            label_train = np.concatenate([label_train, label_temp], axis=0)

    print('classify...')
    acc, f1score = classcify(data_train, label_train, data_test, label_test)

    print('inter sub acc: ', acc)

    acc_all.append(acc)
    f1_all.append(f1score)


print('save...')
scio.savemat('./result/acc_all/acc_DE_NOGED_svm_LOCV.mat',{'acc_all':acc_all,\
'sub_list':np.array(file_list,dtype=np.object)})
    

print('all inter subject acc: ', acc_all)
print('all inter subject mean acc: ', np.mean(np.array(acc_all)))
print('all inter subject std acc: ', np.std(np.array(acc_all)))
