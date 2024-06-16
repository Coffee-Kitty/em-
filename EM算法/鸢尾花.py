from itertools import permutations

import numpy as np
from sklearn import datasets
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import doctest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_iris():

    # ----------导入模块和数据集----------
    import pandas as pd
    from sklearn import datasets
    iris = datasets.load_iris()
    feature_names = iris['feature_names']
    target_names = iris['target_names']
    feature = iris['data']
    labels = iris['target']
    print("特征名:" + str(feature_names))
    print("特征matrix:" + str(feature))
    print("标签名:" + str(target_names))
    print("类别vector:" + str(labels))
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    #
    # iris = datasets.load_iris()
    #
    # # 我们将仅使用花瓣的长度和宽度进行此分析
    # X = iris.data[:, [2, 3]]
    # y = iris.target
    #
    # # 将鸢尾花数据放入pandas DataFrame中
    # iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])
    # # ----------分割数据集----------
    # # 将数据集分为训练和测试数据集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # # ----------数据预处理：标准化数据----------
    # from sklearn.preprocessing import StandardScaler

    # X_scaled = (X - X.mean()) / X.std()
    # 其中, X.mean()为数据集的均值, X.std()为数据集的标准差
    # sc = StandardScaler()
    #
    # sc.fit(feature)
    # feature = sc.transform(feature)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    #
    #
    # X_test_std = sc.transform(X_test)
    return feature, labels





class KMeans:
    def __init__(self, n_clusters, max_iter=50, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def _initialize_centroids_plusplus(self, X):
        """
        使用 K-means++ 初始化质心
        :param X: 输入数据
        :return: 初始化后的质心
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # 随机选择第一个质心
        centroids[0] = X[np.random.randint(n_samples)]

        # 选择剩余的质心
        for i in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - centroids[j], axis=1) for j in range(i)], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = X[j]
                    break

        return centroids

    def fit(self, X, init='random'):
        if init == 'k-means++':
            self.centroids = self._initialize_centroids_plusplus(X)
        else:
            # 随机初始化聚类中心
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # 计算每个样本点到聚类中心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

            # 分配样本到最近的聚类中心
            labels = np.argmin(distances, axis=0)

            # 更新聚类中心
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # 计算聚类中心的变化量
            centroid_change = np.sqrt(((new_centroids - self.centroids) ** 2).sum(axis=1))

            # 更新聚类中心
            self.centroids = new_centroids

            # 判断是否达到收敛条件
            if np.all(centroid_change < self.tol):
                break

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


class EM:
    def __init__(self, n_components, supervised=True,target_labels=None, max_iter=100):
        self.max_iter = max_iter
        self.supervised = supervised  # 是否是纯粹无监督
        self.n_components = n_components
        self.weights = None
        self.mean = None
        self.cov = None

        target_dict = {}
        order = []
        for i in range(n_components):
            target_dict[i] = []
        for i in range(len(target_labels)):
            if target_dict[target_labels[i]] == []:
                order.append(target_labels[i])
            target_dict[target_labels[i]].append(i)

        # 先把order[0] 全部变成0
        # 再把order[1]全部变1
        # 然后 order[2]都为2
        count = 0
        for ord in order:
            for idx in target_dict[ord]:
                target_labels[idx] = count
            count += 1
        self.target_labels = target_labels
        print("self.target:"+str(self.target_labels))

    @staticmethod
    def gaussian(x, mu, sigma):
        """
        一维高斯分布函数
        :param x:
        :param mu: 均值
        :param sigma: 标准差
        :return: 对x做高斯分布后的结果y
        示例：
        # >>> x = np.array([0, 1, 2, 3, 4])
        # >>> mu = 2
        # >>> sigma = 1
        # >>> EM.gaussian(x, mu, sigma)
        array([0.05399097, 0.24197072, 0.39894228, 0.24197072, 0.05399097])
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def mutil_gaussian(X, Mu, Cov):
        """
        多维高斯分布函数
        :param X:
        :param Mu:
        :param Cov: 注意协方差矩阵 需 非奇异
        :return:
        示例：
        # >>> X = np.array([0, 0, 0])  # 要计算的取值
        # >>> Mu = np.array([0, 0, 0])  # 均值向量
        # >>> Cov = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])  # 协方差矩阵
        # >>> EM.mutil_gaussian(X, Mu, Cov)
        0.07699734338570252
        """
        k = len(X)
        covdet = np.linalg.det(Cov)  # 计算|cov|
        covinv = np.linalg.pinv(Cov)  # 计算cov的逆
        if covdet <= 1e-6:  # 以防行列式为0
            covdet = np.linalg.det(Cov + np.eye(k) * 0.01)
            covinv = np.linalg.pinv(Cov + np.eye(k) * 0.01)
        coeff = 1 / np.sqrt((2 * np.pi) ** k * covdet)
        exponent = -0.5 * np.dot(np.dot((X - Mu), covinv), (X - Mu).T)
        return coeff * np.exp(exponent)

    @staticmethod
    def compute_error(n_components, labels, predict_labels):
        """
        将predict_labels 按顺序类别改为  0 1 2
        然后对比
        """
        assert len(labels) > 0 and len(labels) == len(predict_labels)

        nums_dict = {}
        order = []
        for i in range(n_components):
            nums_dict[i] = []
        for i in range(len(predict_labels)):
            if nums_dict[predict_labels[i]] == []:
                order.append(predict_labels[i])
            nums_dict[predict_labels[i]].append(i)

        # 先把order[0] 全部变成0   全排列
        # 再把order[1]全部变1
        # 然后 order[2]都为2
        # 定义要进行全排列的元素列表
        elements = [i for i in range(n_components)]
        # 计算元素列表的全排列
        perms = permutations(elements)
        max_match=0
        for perm in perms:
            # print(perm)
            cidx=0
            for ord in order:
                count = perm[cidx]
                for idx in nums_dict[ord]:
                    predict_labels[idx] = count
                cidx+=1

            match = 0
            for i in range(len(labels)):
                if labels[i] == predict_labels[i]:
                    match += 1
            max_match=max(match,max_match)
        return max_match / len(labels)


    def estep_init(self, features):
        """
        random_from_data 初始化
        :param features:
        :return:
        """
        n = features.shape[0]
        k = self.n_components
        self.mean = features[np.random.choice(n, k, replace=False)]  # 从数据中随机选择 k 个样本作为初始均值向量

        self.cov = np.array([np.eye(features.shape[1]) for _ in range(k)])  # 初始化为单位矩阵

        # weights = np.random.randint(low=0,high=2*self.n_components+1,size=self.n_components)
        # self.weights = weights/ np.sum(weights)  # 初始化为均匀分布
        self.weights = np.ones(self.n_components) / self.n_components  # 初始化为均匀权重

    def mstep_init(self, features):
        """
        random 初始化
        :param features:
        :return:
        """
        # self.weights = np.ones(self.n_components) / self.n_components  # 初始化为均匀权重
        # # 计算均值向量
        # self.mean = np.zeros((self.n_components, features.shape[1]))
        # for i in range(self.n_components):
        #     self.mean[i] = np.mean(features, axis=0)
        # # 计算协方差矩阵
        # self.cov = np.zeros((self.n_components, features.shape[1], features.shape[1]))
        # for i in range(self.n_components):
        #     self.cov[i] = np.cov(features, rowvar=False)

        # 随机初始化z
        predict_labels = np.random.randint(0, self.n_components, size=features.shape[0])
        # 获取类别的数量
        k = len(np.unique(predict_labels))
        # 创建一个与 responsibility 相同形状的零矩阵
        n = len(predict_labels)
        responsibility = np.zeros((n, k))
        # 将每个数据点的类别对应的责任值设置为 1
        for i, label in enumerate(predict_labels):
            responsibility[i, label] = 1
        self.m_step(features, responsibility)

    def kmean_init(self, features):
        """
        kmean 初始化
        :param features:
        :return:
        """
        kmeans = KMeans(n_clusters=self.n_components)
        kmeans.fit(features,init='random')
        self.mean = kmeans.centroids

        # 假设协方差矩阵相同，可以取所有类的样本点的协方差的均值
        self.cov = np.zeros((self.n_components, features.shape[1], features.shape[1]))
        predict_labels = kmeans.predict(features)
        for j in range(self.n_components):
            if len(features[predict_labels == j]) > 1:
                self.cov[j] = np.cov(features[predict_labels == j], rowvar=False).mean(axis=0)
            else:
                self.cov[j] = np.cov(features[predict_labels == j], rowvar=False).mean()
        # 随机初始化z
        predict_labels = kmeans.predict(features)
        # 获取类别的数量
        k = len(np.unique(predict_labels))
        # 创建一个与 responsibility 相同形状的零矩阵
        n = len(predict_labels)
        responsibility = np.zeros((n, k), dtype=np.float64)
        # 将每个数据点的类别对应的责任值设置为 1
        for i, label in enumerate(predict_labels):
            responsibility[i, label] = 1
        self.m_step(features, responsibility)

    def e_step(self, features):
        """
        依据高斯混合模型，计算出每个feature归属类别的向量
        features: n*k 的矩阵，共 n 条数据，每条数据有 k 个维度
        :return:  responsibilities: n*k 的矩阵，每行代表一个数据点，每列代表一个成分的归一化责任值
        """
        n = len(features)
        k = self.n_components
        responsibilities = np.zeros((n, k))
        for i in range(n):
            normalization_factor = \
                sum(self.weights[l] * self.mutil_gaussian(features[i], self.mean[l], self.cov[l]) for l in range(k))
            for j in range(k):
                posterior = self.weights[j] * self.mutil_gaussian(features[i], self.mean[j], self.cov[j])
                responsibilities[i, j] = posterior / normalization_factor
        return responsibilities

    def m_step(self, features, responsibilities):
        """
        最大化似然函数 更新模型参数
        根据新的responsibility 更新 mean cov weights
        :return:
        """
        n = len(features)
        k = self.n_components

        # 更新权重
        self.weights = np.sum(responsibilities, axis=0) / n

        # 更新均值向量
        self.mean = np.dot(responsibilities.T, features) / np.sum(responsibilities, axis=0)[:, np.newaxis]

        # 更新协方差矩阵
        self.cov = np.zeros((k, features.shape[1], features.shape[1]))
        for j in range(k):
            diff = features - self.mean[j]
            self.cov[j] = np.dot(responsibilities[:, j] * diff.T, diff) / np.sum(responsibilities[:, j])

    def print_parameters(self):
        str_param = str()
        for idx, (weight, mean, cov) in enumerate(zip(self.weights, self.mean, self.cov)):
            # print(
                str_param+=("\n"+str(idx) + "\n权重:" + str(weight) + "\n均值:" + str(mean) + "\n协方差" + str(cov))
            # )
        str_param+="\n"
        print(str_param)
        return str_param

    def fit_predict(self, features, epsilon=1e-6,print_count=50,init_Method=None):
        """
        均值向量中 有一项前后误差小于epsilon则终止迭代
        :param features:  n*k的矩阵  共n条数据，每条数据有k个维度
        :return:
        """
        if init_Method is None:
            init_Method=self.kmean_init
        init_Method(features)
        pre_means = None
        count = 0

        str_show_iterations = ""
        try:
            while True:

                responsibility = self.e_step(features)
                # print(f"count:{count}, mean:{self.mean}， cov:{self.cov}")
                # print(f"第{count}轮次：responsibility={responsibility}")
                predict_labels = np.argmax(responsibility, axis=1)  # 获取每行最大值的索引，即分类结果
                # print(f"predict={predict_labels}")

                if count % print_count == 0:

                    # print(f"预测得分为：{prediction_scores}")
                    print(f"第{count}轮次")
                    # print(predict_labels)
                    if self.supervised:
                        errors = self.compute_error(self.n_components, self.target_labels, predict_labels)
                        print(errors)
                        str_show_iterations += f"count:{count}\t errors:{errors}\n"

                    # self.print_parameters()
                    print("*" * 50)
                    # print(predict_labels)
                    # print(labels)

                # 检查收敛条件
                if count >= self.max_iter \
                        or (pre_means is not None and np.min(np.abs(pre_means - self.mean) < epsilon)):
                    print(f"最终迭代轮次count:{count}")
                    self.print_parameters()
                    if self.supervised:
                        errors = self.compute_error(self.n_components, self.target_labels, predict_labels)
                        print(errors)
                    break
                count += 1
                pre_means = self.mean

                self.m_step(features, responsibility)
        except KeyboardInterrupt:
            print("最终情况")
            self.print_parameters()
            return

        responsibility = self.e_step(features)
        predict_labels = np.argmax(responsibility, axis=1)  # 获取每行最大值的索引，即分类结果
        return predict_labels, str_show_iterations


def standardEM(feature, labels):
    gmm = GaussianMixture(n_components=3, covariance_type='full',init_params="random")
    gmm.fit(feature, y=None)
    prediction_labels = gmm.predict(feature)
    print("原始类别:" + str(labels))
    print("聚类预测类别:" + str(prediction_labels))

    scores = calinski_harabasz_score(feature, labels)
    print("原始分数:" + str(scores))
    prediction_scores = calinski_harabasz_score(feature, prediction_labels)
    print("聚类预测分数" + str(prediction_scores))

    error = EM.compute_error(3, labels, prediction_labels)
    print("聚类预测error" + str(error))


def myEM(feature, labels):
    em = EM(n_components=3,max_iter=500,target_labels=labels,supervised=True)
    em.fit_predict(feature,init_Method=em.estep_init)


if __name__ == "__main__":
    # doctest.testmod()
    feature, labels = get_iris()
    for i in range(10):
        feature1, labels1 = get_iris()
        feature = np.concatenate((feature, feature1))
        labels = np.concatenate((labels, labels1))
    # standardEM(f3, l3)

    myEM(feature, labels)
