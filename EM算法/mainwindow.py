import re

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, \
    QTextEdit, QMessageBox, QInputDialog

import numpy as np

import sys

from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
from sklearn import datasets

import 身高体重
import 鸢尾花
from 鸢尾花 import EM


class MainWindow(QMainWindow):

    @staticmethod
    def show_iterations(iterations:str):
        # 正则表达式提取迭代次数和错误率
        pattern = r'count:(\d+)\s+errors:(\d+\.\d+)'
        matches = re.findall(pattern, iterations)

        # 提取数据并转换为适当的格式
        iterations = [int(match[0]) for match in matches]
        errors = [float(match[1]) for match in matches]

        # 绘制图表
        plt.plot(iterations, errors, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Errors')
        plt.title('Iteration vs. Errors')
        plt.grid(True)
        plt.show()

    def __init__(self):
        super().__init__()
        self.features = None
        self.labels = None
        self.init_method = None
        self.n_components = None
        self.em = None
        self.init_main_window()

    def init_main_window(self):
        self.setWindowTitle("EM Algorithm")
        self.setGeometry(400, 400, 800, 800)

        # 创建主窗口部件和布局
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # 展示原始数据
        self.show_origin_data_button = QPushButton("show origin dataset", self)
        self.show_origin_data_button.clicked.connect(self.show_origin_data)
        self.layout.addWidget(self.show_origin_data_button)

        # 1.展示数据集
        self.show_data_button = QPushButton("show dataset", self)
        self.show_data_button.clicked.connect(self.show_data)
        self.layout.addWidget(self.show_data_button)

        # 2.选择初始化 包括kmean random  random_from_data
        self.init_em_button = QPushButton("select init method", self)
        self.init_em_button.clicked.connect(self.init_em)
        self.layout.addWidget(self.init_em_button)

        # 3.执行算法
        self.em_run_button = QPushButton("run em", self)
        self.em_run_button.clicked.connect(self.em_run)
        self.layout.addWidget(self.em_run_button)

        # 创建显示结果的文本框
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        # 调整窗口大小
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

    def show_origin_data(self):
        # 显示输入对话框
        init_datas = ['鸢尾花', '男女身高体重']
        method, ok_pressed = QInputDialog.getItem(self, "选择数据",
                                                  "请选择数据（男女身高体重(10000条)；鸢尾花（150条））:",
                                                  init_datas, 0, False)
        if ok_pressed:
            features=None
            labels = None
            feature_names,target_names = None,None
            if method == '鸢尾花':
                iris = datasets.load_iris()
                feature_names = iris['feature_names']
                target_names = ['花的类别']
                features = iris['data']
                labels = iris['target']
            elif method == '男女身高体重':
                filename = 'data/heights_weights_genders.csv'
                # reading the file
                data_ori = np.genfromtxt(filename, dtype=str, delimiter=',', skip_header=True, encoding='utf-8')
                np.random.shuffle(data_ori)
                feature_names = ['身高','体重']
                target_names = ['性别']
                labels = data_ori[:, 0]
                features = data_ori[:, 1:]

            # 将加载的数据显示在文本框中
            if features is not None and labels is not None:
                max_widths = [max(len(str(f))+2 for f in feature_column) for feature_column in features.T]

                max_label_width = max(len(str(label)) for label in labels)+2


                self.result_text.clear()
                # self.result_text.append("Features\tLabels:\n")
                # 首先展示列名
                column_format = "{:<" + str(max_widths[0]) + "} "
                for i in range(1, len(feature_names)):
                    column_format += "{:<" + str(max_widths[i]) + "} "
                # column_format += "{:<" + str(max_label_width) + "}\n"
                self.result_text.append("idx\t"+column_format.format(*feature_names)+"\t"+str(target_names[0]))

                # 添加分隔线（可选）
                self.result_text.append("-" * (sum(max_widths) + max_label_width + 3 * (len(feature_names) + 1)) + "\n")

                for idx, (feature, label) in enumerate(zip(features, labels)):
                    # 假设特征是列表或数组，将其转换为字符串
                    row_format = "{:<" + str(max_widths[0]) + "} "
                    for i in range(1, len(feature_names)):
                        row_format += "{:<" + str(max_widths[i]) + "} "
                    # row_format += "{:<" + str(max_label_width) + "}"
                    self.result_text.append(f"{idx}:\t"+row_format.format(*feature)+"\t"+str(label))
                print("数据展示完毕")


    def show_data(self):
        # 显示输入对话框
        init_datas = ['鸢尾花', '男女身高体重']
        method, ok_pressed = QInputDialog.getItem(self, "选择数据",
                                                  "请选择数据（男女身高体重(10000条)；鸢尾花（150条））:",
                                                  init_datas, 0, False)
        if ok_pressed:
            if method == '鸢尾花':
                self.features, self.labels = 鸢尾花.get_iris()
                self.n_components = 3
                self.em = EM(n_components=self.n_components, supervised=True, target_labels=self.labels, max_iter=300)
            elif method == '男女身高体重':
                self.features, self.labels = 身高体重.get_heights_weights_datas()
                self.n_components = 2
                self.em = EM(n_components=self.n_components, supervised=True, target_labels=self.labels, max_iter=300)

            # 将加载的数据显示在文本框中
            if self.features is not None and self.labels is not None:
                self.result_text.clear()
                self.result_text.append("Features  Labels:\n")
                for idx,(feature,label) in enumerate(zip(self.features,self.labels)):
                    self.result_text.append(str(idx)+":\t"+str(feature)+"\t"+str(label))
                print("数据展示完毕")

    def init_em(self):
        # 显示输入对话框
        init_methods = ['kmean', 'random', 'random_from_data']
        method, ok_pressed = QInputDialog.getItem(self, "选择初始化方法",
                                                  "请选择初始化方法:",
                                                  init_methods, 0, False)
        if ok_pressed:
            self.result_text.clear()
            if method == 'kmean':
                self.init_method = self.em.kmean_init
                self.result_text.append("init_method: kmean\n")
            elif method == 'random':
                self.init_method = self.em.mstep_init
                self.result_text.append("init_method: random\n")
            elif method == 'random_from_data':
                self.init_method = self.em.estep_init
                self.result_text.append("init_method: random_from_data\n")
            print("初始化方法选择完毕")


    def em_run(self):
        predict_labels, str_iterations = self.em.fit_predict(print_count=1,features=self.features, init_Method=self.init_method)
        self.result_text.clear()
        self.result_text.append("Results:\n")
        self.result_text.append(str(predict_labels))
        self.result_text.append("\nParameters:")
        str_params = self.em.print_parameters()
        self.result_text.append(str_params)
        self.result_text.append("\nAccuracy:")
        errors = EM.compute_error(n_components=self.n_components, labels=self.labels, predict_labels=predict_labels)
        self.result_text.append(str(errors))
        self.result_text.append("\nIteratons:")
        self.result_text.append(str_iterations)
        self.result_text.ensureCursorVisible()  # 确保光标可见
        print("em算法运行完成")

        MainWindow.show_iterations(str_iterations)





if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MainWindow()

    main_window.show()
    app.exec_()
