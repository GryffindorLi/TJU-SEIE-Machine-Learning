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
python main.py [--task] [--prekernel] [--preplot] [--preprocessing] [--cls_method] [--cls_kernel] 
[--plotresult] [--lr] [--alpha] [--optimizer] [--iter] [--M] [--Error]
```

其中，各个参数的定义如下所示：

```python
    parser.add_argument('--task', default='cls', type=str, 
                        help = "choose the task you'd like to perform from 'cls', 'cluster', 'prep'.")
    parser.add_argument('--prekernel', default='rbf', type=str, 
                        help="kernel function used in preprocess")
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
```



## TODO:

完善README.md