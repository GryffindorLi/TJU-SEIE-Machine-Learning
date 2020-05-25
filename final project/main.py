# coding=utf-8
'''
函数主接口，可调用各种模块
@author: LZRsuper666
'''

import argparse
from sklearn import datasets
from data_preprocess import use_PCA, use_KPCA, use_LDA
from classifier import SVM_cls, NN_cls, Logistic_Regression_cls
from FCM import FCM

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cls', type=str, 
                        help = "choose the task you'd like to perform from 'cls', 'cluster', 'prep'.")
    parser.add_argument('--prekernel', default='rbf', type=str, help="kernel function used in preprocess")
    parser.add_argument('--preplot', default=False, type=bool, help='whether plot the results')
    parser.add_argument('--preprocessing', default='PCA', type=str, help='data preprocessing method')
    parser.add_argument('--cls_method', default='SVM', type=str, help='classification method.')
    parser.add_argument('--cls_kernel', default='rbf', type=str, help='SVM kernel function.')
    parser.add_argument('--plotresult', default=True, type=bool, help='whether plot the result.')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for neural network.')
    parser.add_argument('--alpha', default=0.00001, type = float,
                         help='penalty for regularization in neural network.')
    parser.add_argument('--optimizer', default='adam', type = str, 
                        help='optimization method for neural network.')
    parser.add_argument('--iter', default=1000, type = int, 
                        help='number of iteration in neural network and FCM.')
    parser.add_argument('--M', default=1.5, type=float, help='M parameter in FCM')
    parser.add_argument('--Error', default=0.005, type=float, help='error threshold in FCM')

    return parser.parse_args()

if __name__ == '__main__':
    parser = argument()
    if parser.task == 'preprocess':
        if parser.preprocessing == 'PCA':
            X, y = use_PCA('iris_data.txt', plot = parser.preplot)
        elif parser.preprocessing == 'KPCA':
            X, y = use_KPCA('iris_data.txt', plot = parser.preplot, kernel = parser.prekernel)
        elif parser.preprocessing == 'LDA':
            X, y = use_LDA('iris_data.txt', plot = parser.preplot)
        elif parser.preprocessing == 'None':
            loader = datasets.load_iris()
            X, y = loader['data'], loader['target']
        else:
            raise Exception("Wrong data preprocessing method.")
    elif parser.task == 'cls':
        if parser.cls_method == 'SVM':
            p, a = SVM_cls(preprocessing=parser.preprocessing, pre_kernel=parser.prekernel, 
            cls_kernel=parser.cls_kernel, plot_result=parser.plot_result)
        elif parser.cls_method == 'NN':
            p,a = NN_cls(preprocessing=parser.preprocessing, pre_kernel=parser.prekernel,
            lr=parser.lr, a = parser.alpha, optimizer=parser.optimizer, num_iter=parser.iter,
            plot_result=parser.plot_result)
        elif parser.cls_method == 'LR':
            p,a = Logistic_Regression_cls(preprocessing=parser.preprocessing, pre_kernel=parser.prekernel,
            plot_result=parser.plot_result)
        else:
            raise Exception("Wrong classification method!")
    elif parser.task == 'cluster':
        FCM(preprocessing=parser.preprocessing, M = parser.M, Error=parser.Error, Maxiter=parser.iter,
        pre_kernel=parser.prekernel)
    else:
        raise Exception("Wrong task to perform!")

