# 结课项目：鸢尾花数据的预处理、聚类及分类
## 一、环境配置
检查操作系统中是否安装Python3.7。若无安装，请先安装Python3.7。
打开命令行工具，切换工作文件夹到当前目录，输入如下命令：

```
pip install -r requirements.txt
```

随后按照步骤进行操作，完成安装。

## 二、代码运行

在dataloader.py中定义了一个数据加载类；data_preprocess.py是数据预处理模块；FCM.py是模糊均值聚类文件；classifier.py是分类器文件，里面提供了支持向量机（SVM）、logistic regression和神经网络（Neural Network)进行分类。main.py可以调用全部的文件，实现整个系统。main.py的调用格式如下：

```
python main.py [--args1] [--args2] ...
```

## TODO:

完成main.py，完善README.md