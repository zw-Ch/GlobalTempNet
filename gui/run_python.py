import sys
import os
import os.path as osp
import matplotlib
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets


class VisualWidget(QWidget):
    def __init__(self):
        super(VisualWidget, self).__init__()
        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', 'd')
        self.graph_wg_ori = pg.PlotWidget()
        self.graph_wg_result = pg.PlotWidget()
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.graph_wg_ori)
        vbox.addWidget(self.graph_wg_result)

    def plot_ori(self, ts_address):
        x = np.load(ts_address + ".npy")
        self.graph_wg_ori.clear()
        pen = pg.mkPen(color='k', width=2)
        self.graph_wg_ori.plot(x, pen=pen)
        self.set_style(ts_address)

    def plot_result(self, folder, ts_name, model_, model):
        result_address = osp.join(folder, "result", ts_name, model_)
        train_true_address = osp.join(result_address, "train_true_{}.npy".format(model))
        train_predict_address = osp.join(result_address, "train_predict_{}.npy".format(model))
        test_true_address = osp.join(result_address, "test_true_{}.npy".format(model))
        test_predict_address = osp.join(result_address, "test_predict_{}.npy".format(model))

        train_true = np.load(train_true_address)
        train_predict = np.load(train_predict_address)
        test_true = np.load(test_true_address)
        test_predict = np.load(test_predict_address)
        train_range = np.arange(train_true.shape[0])
        test_range = np.arange(train_true.shape[0], train_true.shape[0] + test_true.shape[0])

        self.graph_wg_result.clear()
        self.graph_wg_result.addLegend()
        pen = pg.mkPen(color='m', width=1)
        line1 = self.graph_wg_result.plot(train_range, train_true, pen=pen, name="Train True")
        pen = pg.mkPen(color='b', width=1)
        line2 = self.graph_wg_result.plot(train_range, train_predict, pen=pen, name="Train Predict")
        pen = pg.mkPen(color='g', width=1)
        line3 = self.graph_wg_result.plot(test_range, test_true, pen=pen, name="Test True")
        pen = pg.mkPen(color='r', width=1)
        line4 = self.graph_wg_result.plot(test_range, test_predict, pen=pen, name="Test Predict")

        line1.setAlpha(0.7, False)
        line2.setAlpha(0.99, False)
        line3.setAlpha(0.7, False)
        line4.setAlpha(0.99, False)

    def set_style(self, ts_address):
        ts_address_ = ts_address.split('/')
        ts = ts_address_[-2]
        if ts == 'HadCRUT5':
            self.graph_wg_ori.setLabel('left', 'Anomaly (deg C)')
            # self.graph_wg_ori.setLabel('bottom', 'year')
            self.graph_wg_result.setLabel('left', 'Anomaly (deg C)')
            # self.graph_wg_result.setLabel('bottom', 'year')
        elif ts == 'cli_dash':
            self.graph_wg_ori.setLabel('left', 'Anomaly (deg C)')
            self.graph_wg_result.setLabel('left', 'Anomaly (deg C)')
        elif ts == 'temp_month':
            self.graph_wg_ori.setLabel('left', 'Temperature')
            self.graph_wg_result.setLabel('left', 'Temperature')
        elif ts == 'elect':
            self.graph_wg_ori.setLabel('left', 'Electricity')
            self.graph_wg_result.setLabel('left', 'Electricity')
        elif ts == 'traffic':
            self.graph_wg_ori.setLabel('left', 'Traffic')
            self.graph_wg_result.setLabel('left', 'Traffic')
        elif ts == 'sales':
            self.graph_wg_ori.setLabel('left', 'Exchange-Rate')
            self.graph_wg_result.setLabel('left', 'Exchange-Rate')
        else:
            raise TypeError("Unknown Type of folder!")
        self.graph_wg_ori.plotItem.setTitle('Original Data')
        self.graph_wg_result.plotItem.setTitle('Predicted Results')


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.init()

    def init(self):
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(self.exit_act())
        file_menu.addAction(self.run_act())

        self.ts_name = QLabel('ts name')
        self.ts_folder = QLabel('ts folder')
        self.epochs = QLabel('epochs')
        self.model = QLabel('model')
        self.root = QLabel('root')
        self.ts_address = QLabel('ts address')

        self.ts_folder_cb = self.get_ts_folder_cb()
        self.ts_name_cb = self.get_ts_name_cb()
        self.ts_folder_cb.currentTextChanged.connect(self.change_ts_name)
        self.epochs_edit = QLineEdit('1000')
        self.model_cb = self.get_model_cb()
        self.root_edit = QLineEdit('/home/chenziwei2021/py_envs_pc/paper/ResGraphNet/datasets')
        self.ts_address_edit = QLineEdit()
        self.ts_address_button = self.get_ts_address_button()
        self.model_button = self.get_model_button()
        self.vis_wg = VisualWidget()
        self.ts_address_edit.textChanged.connect(self.vis_wg.plot_ori)

        self.set_layout()
        # self.set_layout_form()

        self.setGeometry(100, 100, 1600, 800)
        self.setWindowTitle('Drawing text')
        self.center()
        self.show()

    def set_layout(self):
        self.grid = QGridLayout()
        self.grid.addWidget(self.ts_folder, 1, 0)
        self.grid.addWidget(self.ts_folder_cb, 1, 1)
        self.grid.addWidget(self.ts_name, 2, 0)
        self.grid.addWidget(self.ts_name_cb, 2, 1)
        self.grid.addWidget(self.root, 3, 0)
        self.grid.addWidget(self.root_edit, 3, 1)
        self.grid.addWidget(self.ts_address, 4, 0)
        self.grid.addWidget(self.ts_address_edit, 4, 1)
        self.grid.addWidget(self.ts_address_button, 4, 2)
        self.grid.addWidget(self.model, 5, 0)
        self.grid.addWidget(self.model_cb, 5, 1)
        self.grid.addWidget(self.model_button, 5, 2)
        self.grid.addWidget(self.epochs, 6, 0)
        self.grid.addWidget(self.epochs_edit, 6, 1)
        hbox = QHBoxLayout()
        hbox.addLayout(self.grid)
        hbox.addWidget(self.vis_wg)

        self.setLayout(hbox)

    def exit_act(self):
        exitAct = QAction(QIcon('toggle2-off.svg'), 'Exit', self)
        exitAct.setShortcut('Ctrl+w')
        exitAct.triggered.connect(qApp.quit)
        return exitAct

    def run_act(self):
        runAct = QAction(QIcon('run.svg'), 'run', self)
        runAct.setShortcut('Ctrl+r')
        runAct.triggered.connect(self.run_program)
        return runAct

    def run_program(self):
        folder = "/home/chenziwei2021/py_envs_pc/paper/ResGraphNet"
        run_folder = osp.join(folder, "run")
        ts_folder = self.ts_folder_cb.currentText()
        ts_name = self.ts_name_cb.currentText()
        model = self.model_cb.currentText()
        epochs = self.epochs_edit.text()
        os.chdir(run_folder)
        if model == 'ResGraphNet':
            run_address = osp.join(run_folder, "run_ResGraphNet.py")
            model_ = model
            command = "python {} {} {} {}".format(run_address, ts_folder, ts_name, epochs)
        elif model == "RESModel":
            run_address = osp.join(run_folder, "run_RESModel.py")
            model_ = model
            command = "python {} {} {} {}".format(run_address, ts_folder, ts_name, epochs)
        elif model in ['forest', 'linear', 'svr', 'sgd']:
            run_address = osp.join(run_folder, "run_MLModel.py")
            model_ = "MLModel"
            ml_style = model
            command = 'python {} {} {} {}'.format(run_address, ts_folder, ts_name, ml_style)
        elif model in ['GraphSage', 'UniMP', 'GCN', 'GIN']:
            run_address = osp.join(run_folder, "run_GNNModel.py")
            model_ = "GNNModel"
            gnn_style = model
            command = "python {} {} {} {} {}".format(run_address, ts_folder, ts_name, gnn_style, epochs)
        elif model in ["LSTM", "GRU"]:
            run_address = osp.join(run_folder, "run_RNNModel.py")
            model_ = "RNNModel"
            rnn_style = model
            command = "python {} {} {} {} {}".format(run_address, ts_folder, ts_name, rnn_style, epochs)
        elif model == "ARIMA":
            run_address = osp.join(run_folder, "run_ARIMA.py")
            model_ = "ARIMA"
            command = "python {} {} {}".format(run_address, ts_folder, ts_name)
        elif model == "SARIMAX":
            run_address = osp.join(run_folder, "run_SARIMAX.py")
            model_ = "SARIMAX"
            command = "python {} {} {}".format(run_address, ts_folder, ts_name)
        else:
            raise TypeError("Unknown Type of model!")
        os.system(command)

        # plot result
        self.vis_wg.plot_result(folder, ts_name, model_, model)

        os.chdir(os.getcwd())

    def get_ts_address_button(self):
        ts_address_button = QPushButton('get')
        ts_address_button.clicked.connect(self.ts_address_click)
        return ts_address_button

    def get_model_button(self):
        model_button = QPushButton('run')
        model_button.clicked.connect(self.run_program)
        return model_button

    def ts_address_click(self):
        ts_folder = self.ts_folder_cb.currentText()
        ts = self.ts_name_cb.currentText()
        root = self.root_edit.text()
        ts_address = osp.join(root, ts_folder, ts)
        self.ts_address_edit.text = ts_address
        self.ts_address_edit.setText(ts_address)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))

    def get_ts_folder_cb(self):
        ts_folder_cb = QComboBox()
        ts_folder_cb.addItems(['HadCRUT5', 'cli_dash', 'temp_month', 'elect', 'traffic', 'sales'])
        return ts_folder_cb

    def get_ts_name_cb(self):
        ts_name_cb = QComboBox()
        ts_name_cb.addItems(['HadCRUT5_global', 'HadCRUT5_northern', 'HadCRUT5_southern'])
        return ts_name_cb

    def get_model_cb(self):
        model_cb = QComboBox()
        model_cb.addItems(['ResGraphNet', 'RESModel', 'GraphSage', 'UniMP', 'GCN', 'GIN', 'forest', 'linear', 'svr',
                           'sgd', 'LSTM', 'GRU', 'ARIMA', 'SARIMAX'])
        return model_cb

    def change_ts_name(self, text):
        ts_folder_all = ['HadCRUT5', 'cli_dash', 'temp_month', 'elect', 'traffic', 'sales']
        ts_HadCRUT5 = ['HadCRUT5_global', 'HadCRUT5_northern', 'HadCRUT5_southern']
        ts_cli_dash = ['Berkeley_Earth', 'ERA5_European', 'ERA5_Global', 'HadSST3']
        ts_temp_month = ['ERSSTv3b', 'ERSSTv4', 'NOAA']
        ts_elect = ['elect']
        ts_traffic = ['traffic']
        ts_sales = ['sales']
        ts_lists = ts_HadCRUT5, ts_cli_dash, ts_temp_month, ts_elect, ts_traffic, ts_sales
        self.ts_name_cb.clear()
        self.ts_name_cb.addItems(ts_lists[ts_folder_all.index(text)])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
