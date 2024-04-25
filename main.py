from Meka2.meka.Calculate import Performance, Parameters, File_merge, Average_single_indicator

"""
读取result_7_I500_1.txt
标签数为7，样本都存在标签，zeros设置为False
"""

# sl = Performance('./merge_samples.txt', 14, zeros=False)
sl = Performance('./my_file/group1_RF_K_2.txt', 4, zeros=True)
"""
计算单标签和多标签指标，生成文件flag=True
"""
# sl.Multi_indicator(flag=True, zeros=True)
# sl.Multi_indicator(flag=True)
sl.Single_indicator(flag=True, estimated=False)
"""
画图，传入标签名列表
"""
"---------------以下的画图为单个文件绘画ROC曲线和PR曲线图----------------------"
"对于参数estimated，若estimated=False则使用精确值计算面积和绘画图形，使用折线图"
"当estimated=True表示使用估计值计算面积和绘画图形，使用阶梯图 "
"关于estimated参数选择，在Meka软件里默认使用估计值计算，但是由于可选择阈值很少，使用估计值计算，画出的阶梯图跨度太大，效果不好"
"因此我这里默认使用精确值计算， 注意使用精确值计算后的PRC和ROC面积与Meka跑出的结果面积是有差距的"
"而当使用估计值计算，此时PRC和ROC面积值和Meka跑出的结果面积是一样大小的，因此这个estimated值选取需要自己衡量"

"注意：在同一个对象里的estimated值必须保持一致"
# sl.Plot_PRC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'], estimated=False, save=True)
# sl.Plot_ROC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'], estimated=False, save=False)
# sl.Plot_PRC(['L1', 'L2', 'L3', 'L4'], estimated=False, save=False)
# sl.Plot_ROC(['L1', 'L2', 'L3', 'L4'], estimated=False, save=False)
#
# "---------------以下的画图为多个文件绘画ROC曲线和PR曲线图的平均值图--------------"
# Asi = Average_single_indicator('Independent test/', 7)
# Asi.Average_Plot_ROC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])
# Asi.Average_Plot_PRC(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])

"""
独立测试集文件合并
生成merge_samples.txt
"""
# fm = File_merge("Independent test2/")
# fm.file_merge()

"""
寻找多个文件的最优模型文件, 指标为Accuracy
"""
# pa = Parameters('Independent test/', norm='Accuracy')
# pa.maxparameters()

