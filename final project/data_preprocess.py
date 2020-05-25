# coding=utf-8
'''
数据预处理，使用PCA, KPCA, LDA, KLDA
@author: LZRsuper666
'''

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from dataloader import dataloader


def use_KPCA(data_file, plot = False, kernel = 'rbf'):
    if kernel not in ['linear','poly','rbf','sigmoid']:
        print('Please choose a kernel from the following kernel functions\n')
        print(['linear','poly','rbf','sigmoid'])
        return
  
    loader = datasets.load_iris()
    data, label = loader['data'], loader['target']

    kpca = KernelPCA(n_components=2, kernel=kernel)
    kpca.fit(data)
    X_r = kpca.transform(data)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for l in np.unique(label):
            position = label == l
            ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%l)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.legend(loc='best')
            ax.set_title('kernel=%s'% kernel)
        plt.suptitle("KPCA")
        plt.show()
    return X_r, label

def use_LDA(data_file, plot = False):
    iris = datasets.load_iris()
    data, label = iris['data'], iris['target']
    model =  LDA(n_components=2)
    model.fit(data, label)
    data_new = model.transform(data)
    if (plot):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(data_new[:, 0], data_new[:, 1],marker='o',c=label)
        ax.set_title('LDA')
        plt.show()
    return data_new, label


#def use_KLDA():
#    return


def use_PCA(data_file, plot = False):
    iris = datasets.load_iris()
    data, label = iris['data'], iris['target']
    model = PCA(n_components=2)
    model.fit(data)
    new_data = model.transform(data)
#    print('explained variance ratio : %s :'% str(model.explained_variance_ratio_))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for l in np.unique(label):
            position = label == l
            ax.scatter(new_data[position,0],new_data[position,1],label='target=%d'%l)
        ax.set_xlabel('X[0]')
        ax.set_ylabel('Y[0]')
        ax.legend(loc='best')
        ax.set_title('PCA')
        plt.show()
    return new_data, label


if __name__ == '__main__':
    new_feat = use_LDA('iris_data.txt', True)
    #new_feat = use_PCA('iris_data.txt', True)
    #new_feat = use_KPCA('iris_data.txt', True)
    #new_feat = use_KPCA('iris_data.txt', True, 'linear')
    #new_feat = use_KLDA()