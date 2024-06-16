import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from 鸢尾花 import EM

def get_heights_weights_datas():
    filename = 'data/heights_weights_genders.csv'
    # reading the file
    data_ori = np.genfromtxt(filename, dtype=str, delimiter=',', skip_header=True, encoding='utf-8')
    np.random.shuffle(data_ori)
    labels = data_ori[:, 0]
    features = data_ori[:, 1:]
    # 将标签中的 "Male" 替换为 0，"Female" 替换为 1
    labels[labels == '"Male"'] = 0
    labels[labels == '"Female"'] = 1
    labels = labels.astype(int)
    # 将特征从字符串转换为浮点数
    features = features.astype(float)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features,labels

if __name__ == '__main__':
    # data_ori = pd.read_csv('data/heights_weights_genders.csv', encoding='utf-8')
    features, labels = get_heights_weights_datas()
    cls = 2
    em = EM(n_components=cls, supervised=True,target_labels=labels,max_iter=500)
    p2 = em.fit_predict(features,print_count=50,init_Method=em.kmean_init)
    res=None
    for p in p2:
        res=p
    p2 = res
    # str_list="[3 3 4 4 3 3 1 4 4 3 4 3 4 3 4 4 3 4 3 0 2 0 0 0 0 0 0 0 1 2 3 1 2 1 2 1 1 2 2 2 2 1 1 2 1 2 3 2 3 1 3 3 3 0 0 3 3 3 0 3 3 3 3 3 3 3 3 3 0]"
    # p2=[int(x) for x in str_list.strip('[]').split()]
    print(p2)
    # gmm = GaussianMixture(n_components=cls, covariance_type='full', max_iter=100)
    # gmm.fit(features)
    # p1 = gmm.predict(features)
    # print(p1)
    # print(EM.compute_error(n_components=2, labels=labels, predict_labels=p1))

