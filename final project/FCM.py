import numpy as np
import scipy
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

from data_preprocess import use_PCA, use_KPCA, use_LDA

def FCM(preprocessing = 'PCA', M = 1.5, Error = 0.005, Maxiter = 1000, pre_kernel = 'rbf'):
    if preprocessing == 'PCA':
        X, y = use_PCA('iris_data.txt')
    elif preprocessing == 'KPCA':
        X, y = use_KPCA('iris_data.txt', kernel = pre_kernel)
    elif preprocessing == 'LDA':
        X, y = use_LDA('iris_data.txt')
    elif preprocessing == 'None':
        loader = datasets.load_iris()
        X, y = loader['data'], loader['target']
    else:
        print('Please choose a data preprocessing method from the following method:\n')
        print('1.PCA, 2.KPCA, 3.LDA, 4.None')
        return
    
    X = X.T
    center, u, u0, d, jm, p, fpc = cmeans(X, m = M, c=3, error = Error, maxiter = Maxiter)

    for i in u:
        label = np.argmax(u, axis=0)

    fig1 = plt.subplot(1,2,1)
    fig1.set_title('Data after preprocessing')
    for i, tag in enumerate(y):
        if tag == 0:
            fig1.scatter(X[0][i], X[1][i], c='r')
        elif tag == 1:
            fig1.scatter(X[0][i], X[1][i], c='g')
        elif tag == 2:
            fig1.scatter(X[0][i], X[1][i], c='b')
    
    fig2 = plt.subplot(1,2,2)
    fig2.set_title('Clustering result')
    for i, label in enumerate(label):
        if label == 0:
            fig2.scatter(X[0][i], X[1][i], c='r')
        elif label == 1:
            fig2.scatter(X[0][i], X[1][i], c='g')
        elif label == 2:
            fig2.scatter(X[0][i], X[1][i], c='b')
 
    plt.show()


if __name__ == '__main__':
    FCM(preprocessing='KPCA', pre_kernel='poly')