# coding=utf-8
'''
分类器，提供SVM，神经网络和logistic regression的接口
@author: LZRsuper666
'''

import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from Mytorch import NeuralNetwork
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

from data_preprocess import use_PCA, use_KPCA, use_LDA

def SVM_cls(preprocessing = 'PCA', pre_kernel = 'rbf', cls_kernel = 'rbf', plot_result = False):
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)
    classifier = svm.SVC(decision_function_shape='ovo', gamma = 'auto', kernel = cls_kernel)
    classifier.fit(X_train, y_train)

    predict = classifier.predict(X_test)

    total = np.size(y_test)
    correct = 0
    
    for index, label in enumerate(y_test):
        if predict[index] == label:
            correct += 1
    accuracy = correct/total
    print("正确样本数为{}, 正确率为{:.4f}".format(correct, accuracy))

    if plot_result and preprocessing != 'None':
        fig1 = plt.subplot(1,2,1)
        fig1.set_title('raw data with label')
        for idx, y in enumerate(y_test):
            if y == 0:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if y == 1:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if y == 2:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='b')
        
        fig2 = plt.subplot(1,2,2)
        fig2.set_title('classification result')
        for idx, label in enumerate(predict):
            if label == 0:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if label == 1:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if label == 2:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='b')
        plt.show()
    
    return predict, accuracy


def Logistic_Regression_cls(preprocessing = 'PCA', pre_kernel = 'rbf', plot_result = False):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)
    classifier = LogisticRegression(multi_class="multinomial", solver="newton-cg")
    classifier.fit(X_train, y_train)

    predict = classifier.predict(X_test)
    total = np.size(y_test)
    correct = 0
    
    for index, label in enumerate(y_test):
        if predict[index] == label:
            correct += 1
    accuracy = correct/total
    print("正确样本数为{}, 正确率为{:.4f}".format(correct, accuracy))

    if plot_result and preprocessing != 'None':
        fig1 = plt.subplot(1,2,1)
        fig1.set_title('raw data with label')
        for idx, y in enumerate(y_test):
            if y == 0:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if y == 1:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if y == 2:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='b')
        
        fig2 = plt.subplot(1,2,2)
        fig2.set_title('classification result')
        for idx, label in enumerate(predict):
            if label == 0:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if label == 1:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if label == 2:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='b')
        plt.show()

    return predict, accuracy

def NN_cls(preprocessing = 'PCA', pre_kernel = 'rbf', lr = 0.0001, a = 0.00001, optimizer = 'adam', \
     num_iter = 1000, plot_result = False):
    if preprocessing == 'PCA':
        X, y = use_PCA('iris_data.txt')
    elif preprocessing == 'KPCA':
        X, y = use_KPCA('iris_data.txt', kernel = pre_kernel)
    elif preprocessing == 'LDA':
        X, y = use_LDA('iris_data.txt')
    else:
        print('Please choose a data preprocessing method from the following method:\n')
        print('1.PCA, 2.KPCA, 3.LDA')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8)

    classifier = MLP(hidden_layer_sizes=(200, 100), solver=optimizer, learning_rate_init=lr,
    max_iter=num_iter,  alpha = a)
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    total = np.size(y_test)
    correct = 0
    
    for index, label in enumerate(y_test):
        if predict[index] == label:
            correct += 1
    accuracy = correct/total
    print("正确样本数为{}, 正确率为{:.4f}".format(correct, accuracy))
    if plot_result:
        fig1 = plt.subplot(1,2,1)
        fig1.set_title('raw data with label')
        for idx, y in enumerate(y_test):
            if y == 0:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if y == 1:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if y == 2:
                fig1.scatter(X_test[idx][0], X_test[idx][1], c='b')
        
        fig2 = plt.subplot(1,2,2)
        fig2.set_title('classification result')
        for idx, label in enumerate(predict):
            if label == 0:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='r')
            if label == 1:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='g')
            if label == 2:
                fig2.scatter(X_test[idx][0], X_test[idx][1], c='b')
        plt.show()

    return predict, accuracy



    
if __name__ == '__main__':
    p, a = NN_cls('PCA', plot_result=True)