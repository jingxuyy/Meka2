# 注意：

​	用记事本打开本文档，格式较差。

​	可安装typora软件后再次打开。

## 2023/6/27 新BUG修复
### 修复了使用Matplotlib画多张图，在使用savefig函数导致图片内容重叠得问题
    进行了图片保存优化，解决了第一张图的内容出现在第二张图片的矛盾现象
## BUG修复
### 修复了在绘画单个文件单标签PRC曲线时，其计算面积和曲线面积对应错误的计算BUG
    此次更新为绘画PRC曲线和ROC曲线，提供了两种模式
    1. estimated=False
        在此模式下，所有的PRC和ROC曲线计算采用的是精确值计算，图形为折线图
    2. estimated=True
        在此模式下，所有的PRC和ROC曲线计算采用的是估计值计算，图形为阶梯图
    注意：Meka软件跑出来的结果是估计值，并非精确值
    关于estimated的选择，当结果阈值较多（这里大概五六个以上吧）选择估计值较好，这样计算的结果和Meka结果
        完全一致，而当你的结果阈值只要一两个，选择精确值，此时使用折线图，尽管计算结果和Meka结果差距较大
        也是可以的。
    另外同一个对象应该保证estimated值相同

## 新增功能
### 新增Average_single_indicator类，用于绘画多个文件的平均ROC和PRC曲线图
    在创建Average_single_indicator对象时需要传入多个文件所在的文件夹路径，和标签类别个数
    Asi = Average_single_indicator('Independent test/', 7)
    Asi.Average_Plot_ROC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])
    Asi.Average_Plot_PRC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])
若想计算多个文件的其他指标，本版本暂不提供统一方法，用户可以依次创建单个文件对象，然后计算其平均值


# 使用教程

## 1. Meka2.0升级说明

### 1.1 修复了在文件转换过程中eval()函数使用BUG

    在Meka1.0中将txt文件数据转换成列表存在异常转换。

### 1.2 修复了画图PRC曲线图函数显示不完整BUG
    在Meka1.0中存在部分PRC曲线无法连接到(1,0)坐标点

### 1.3 解决了1.0版本使用过程中会生成多个中间文件问题
    在1.0版本中存在.txt->.csv->int_.csv文件转换.csv

### 1.4 解决了输入文件需要指定字符串切片位置(参数问题)
    在当前版本不需要人为指定指定字符串切片索引位置，改成程序自动识别切片位置

### 1.5 解决了1.0版本文件路径多次输入问题
    在当前版本，用户只需要输入一次文件地址即可

### 1.6 解决了1.0版本需要各个文件独立运行问题
    2.0版本将各个模块封装成类，只需要创建相应对象即可


## 2. Meka2.0使用说明

### 2.1 文件夹介绍
    初始状态共有三个文件夹分别是：
    1. meka
    2. my_file
    3. Independent test
1. 其中meka文件夹存放在是预留好的计算代码，这个文件夹里最好不要存放其他文件.

2. my_file文件夹是存放在Meka机器学习框架下跑出来. txt文件的结果,在这个文件夹下,使用者可以创建多个子文件夹,用来存放不同模型结果

3. 独立测试文件夹比较特殊,是存放外部测试(手动n折交叉验证)的. txt结果集。存放在这个文件的数据，最终会被创建的对象合并成一个.txt文件使用。

### 2.2 代码介绍

所有代码都封装在Calculate.py文件里，此文件将操作细节封装成三个类，如下:

| 类名              | 说明                              |
| ------------------ | --------------------------------- |
| Performance    | 此类用来计算多标签分类和二分类的各个指标以及绘画ROC和PR曲线 |
| Parameters      | 此类是寻找多个txt文件中指标最高的模型,例如准确性指标或者精确匹配指标,其结果会按照降序生成一个txt文件          |
| File_merge      | 此类继承了参数说明类、实现外部验证或者手动n倍交叉验证生成多个结果文件、合并成一个文件          |


#### 2.2.1 Performance类中主要方法介绍


    1.init方法需要接收三个参数
        __init__(self, filename, label_num, zeros=False)
        在创建对象时，一般情况只需要指定参数filename和label_num
| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| filename          | 路径参数是txt文件的路径(当前路径只需要传递文件名) |
| label_num          | 指的是分类的标签数(int类型)            |
| zeros           | 默认为False，当设置为True时，是为了解决有的实验其测试集或者训练集存在没有标签的样本          |

    2.Single_indicator方法需要接收1个参数
        Single_indicator(self, flag=False)
| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| flag          | 默认为False,此方法是用来计算单标签的Accuracy,Precision,Recall,F1-score,AUROC,AUPRC六个指标. 其最终结果是一个pandas表格形式展示出来，当flag为False时，只在控制台打印，如果flag为True时，则会将表格生成为Single label performance indicators.csv文件，方便统计 |


    3.Multi_indicator方法需要接收1个参数
        Multi_indicator(self, flag=False)
| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| flag          | 默认为False,此方法用来计算多标签Aiming,Coverage,Accuracy,Absolute_true, Absolute_false五个指标，其最终结果是一个pandas表格形式展示出来，当flag为False时，只在控制台打印，如果flag为True时，则会将表格生成为Multi-label performance indicators.csv文件，方便统计|

    4. Plot_ROC和Plot_PRC方法需要接收2个参数
        Plot_ROC(self, label_name, save=False)
        Plot_PRC(self, label_name, save=False)
| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| label_name         | 需要传递一个列出列表，其大小和标签数相同，列表里要存储每个标签的标签名 |
| save          | 默认为False, 说明画出的曲线不保存，若为True则会保存为pdf格式            |


#### 2.2.2 Parameters类中主要方法介绍

    1.init方法需要接收三个参数
        __init__(self, path, same=True, norm='Accuracy')
| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| path          | 文件夹路径，注意需要额外添加'/'，例如：D:/BaiduNetdiskDownload/test/ |
| same          | 表示每个txt文件的样本数是否一样, 默认为True            |
| norm           | 统计标准，默认为Accuracy，也可以选择Exact match，目前只支持这两种统计方式|

    2.maxparameters方法,无参数
    调用此方法后会将传递的文件夹里的.txt文件，按照指定标准统计后从高到低排序，生成result.txt文件

#### 2.2.3 File_merge类中主要方法介绍

    1.init方法需要接收2个参数
        __init__(self, path, same=True)

| 参数               | 说明                              |
| ------------------ | --------------------------------- |
| path         | 存放独立测试集文件夹地址, 注意需要额外添加'/'|
| same          | 表示每个txt文件的样本数是否一样, 默认为True            |

    2.file_merge方法,无参数
        file_merge(self)
        调用此方法后会将所有的独立测试集文件组会和成一个整体文件merge_samples.txt

## 3. Meka2.0使用示例
使用示例在main.py文件，用户可在main.py文件中运行自己的文件