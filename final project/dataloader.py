# coding=utf-8
'''
本文件定义dataloader类，读取并且对数据集进行分割
'''


import numpy as np
import scipy
import pandas as pd
import matplotlib as plt
import sklearn
import random

def transform_label(label):
        list_label = label.values.tolist()
        int_label = []
        for item in list_label:
            print(item)
            if item == 'Iris-setosa':
                int_label.append(0)
            elif item == 'Iris-versicolor':
                int_label.append(1)
            else:
                int_label.append(2)
        new_label = np.array(int_label)
        return new_label

class dataloader:
    #初始化类，is_classification == True时，按分类方法划分数据集，False按聚类方法划分数据集
    def __init__(self, dataset, is_classification):
        self.data = pd.read_csv(dataset, header=None, encoding='utf-8', sep=',', names = ['feat1', 'feat2', 'feat3', 'feat4', 'label'])
        self.cls = is_classification

    
    #分割数据集
    def split(self):
        frame = pd.DataFrame(self.data, columns = ['feat1', 'feat2', 'feat3', 'feat4', 'label'])
        if (not self.cls):
            label = frame['label']
            label = transform_label(label)
            data = frame[['feat1', 'feat2', 'feat3', 'feat4']]
            data = data.values
            return (data, label)
        else:
            #选取测试集
#            choice = np.random.randint(0, 49,size=10)
            choice1 = random.sample(range(0,50),10);
            choice2 = [x + 50 for x in choice1]
            choice3 = [x + 100 for x in choice1]
            test_choice = choice1 + choice2 + choice3
            test_label = frame.loc[test_choice, 'label']
            test_label = transform_label(test_label)
            test_data = frame.loc[test_choice, ['feat1', 'feat2', 'feat3', 'feat4']].values

            #选取训练集
#            print(len(train_choice))
            train_choice = []
            for y in range(150):
                if y not in test_choice:
                    train_choice.append(y)
            train_label = frame.loc[train_choice, 'label']
            train_label = transform_label(train_label)
            train_data = frame.loc[train_choice, ['feat1', 'feat2', 'feat3', 'feat4']].values
            return (train_data, train_label, test_data, test_label)
    


if __name__ == '__main__':
    loader = dataloader('iris_data.txt', is_classification = True)
    training_data, training_label, testing_data, testing_label = loader.split()
    print(training_label)
    