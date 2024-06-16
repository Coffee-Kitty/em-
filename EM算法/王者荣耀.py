# -*- coding:utf-8 -*-
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
# 使用GMM高斯混合模型进行聚类
from sklearn.mixture import GaussianMixture
# 使用标准化正态分布进行数据规范化
from sklearn.preprocessing import StandardScaler

from 鸢尾花 import EM

def show_data(data_ori):
    # 特征值选择
    features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力',u'法力成长',u'初始法力',u'最高物攻',
                u'物攻成长',u'初始物攻',u'最大物防',u'物防成长',u'初始物防',u'最大每5秒回血',u'每5秒回血成长',
                u'初始每5秒回血',u'最大每5秒回蓝',u'每5秒回蓝成长',u'初始每5秒回蓝',u'最大攻速',u'攻击范围']
    # 选取特征值数据
    data = data_ori[features]
    data.loc[:, u'最大攻速'] = data[u'最大攻速'].apply(lambda x: float(x.strip('%')) / 100)
    data.loc[:, u'攻击范围'] = data[u'攻击范围'].map({'远程': 1, '近战': 0})

    # 对英雄属性之间的关系进行可视化
    # 设置plt正确显示中文，
    plt.rcParams['font.sans-serif'] = ['simhei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # 使用热力图来进行显示features_mean 字段之间的相关性
    corr = data[features].corr()
    plt.figure(figsize=(14,14))
    # annot=True 显示每个方格的数据
    sns.heatmap(corr, annot=True)
    plt.show()



def process_data(data_ori):
    # 相关性大的属性保留一个，因此可以对属性进行降维
    features_remain = [u'最大生命',u'初始生命',u'最大法力',u'最高物攻',
                u'初始物攻',u'最大物防',u'初始物防',u'最大每5秒回血',
                u'最大每5秒回蓝',u'初始每5秒回蓝',u'最大攻速',u'攻击范围']
    data = data_ori[features_remain]
    # 处理最大攻速的百分号和攻击范围的近程和远程
    data.loc[:, u'最大攻速'] = data[u'最大攻速'].apply(lambda x: float(x.strip('%')) / 100)
    data.loc[:, u'攻击范围'] = data[u'攻击范围'].map({'远程': 1, '近战': 0})

    # 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
    ss = StandardScaler()
    data = ss.fit_transform(data)
def standard():
    #构造GMM聚类
    gmm=GaussianMixture(n_components=30,covariance_type='full')
    gmm.fit(data)
    #训练数据
    prediction=gmm.predict(data)
    print(prediction)
    # #将结果输出到CSV文件
    # data_ori.insert(0,'分组',prediction)
    # data_ori.to_csv('./standard_hero_out.csv',index=False,sep=',')
    return prediction

def myEM():
    em = EM(n_components=30,supervised=False)
    prediction=em.fit_predict(data)
    print(prediction)
    # # 将结果输出到CSV文件
    # data_ori.insert(0, '分组', prediction)
    # data_ori.to_csv('./myEm_hero_out.csv', index=False, sep=',')
    return prediction

def printf_result(p,n_components):
    clusters = [[] for i in range(n_components)]
    for idx, cls in enumerate(p):
        # print(f"idx:{idx} cls:{cls}")
        clusters[cls].append(data_ori.loc[idx, u'英雄'])
    for i in range(n_components):
        print(clusters[i])

if __name__ =='__main__':
    # p1=standard()
    # p2=myEM()
    # errors = []
    # for i in range(2,30,5):
    #     gmm = GaussianMixture(n_components=i, covariance_type='full',max_iter=500)
    #     gmm.fit(data)
    #     p1 = gmm.predict(data)
    #     print(p1)
    #     em = EM(n_components=i, supervised=False)
    #     p2 = em.fit_predict(data)
    #     print(p2)
    #     error = EM.compute_error(n_components=i,labels=p1,predict_labels=p2)
    #     print(error)
    #     errors.append({i,error})
    # print(errors)
    # 数据加载，避免中文乱码问题
    data_ori = pd.read_csv('data/heros.csv', encoding='gb18030')
    data = process_data(data_ori)
    cls = 6
    em = EM(n_components=cls, supervised=False)
    p2 = em.fit_predict(data)
    # str_list="[3 3 4 4 3 3 1 4 4 3 4 3 4 3 4 4 3 4 3 0 2 0 0 0 0 0 0 0 1 2 3 1 2 1 2 1 1 2 2 2 2 1 1 2 1 2 3 2 3 1 3 3 3 0 0 3 3 3 0 3 3 3 3 3 3 3 3 3 0]"
    # p2=[int(x) for x in str_list.strip('[]').split()]
    print(p2)
    printf_result(p2,cls)
    gmm = GaussianMixture(n_components=cls, covariance_type='full', max_iter=500)
    gmm.fit(data)
    p1 = gmm.predict(data)
    print(p1)
    printf_result(p1, cls)