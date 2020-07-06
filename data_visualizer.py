import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import logging_config as log
from utils.data_utils import DataUtils


class DataVisualizer:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    """
    데이터를 시각화 한다.
    """

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)
        sns.set_style("whitegrid")
        # plt.rc('font', family="Malgun Gothic")

    def draw_line_chart(self, data: pd.DataFrame, save_filename: str = None, show: bool = False):
        """
        선 그래프를 그린다.
        :param data: 데이터
        :param save_filename: 저장하는 파일명
        :param show: 보여주는지 여부
        :return:
        """
        plt.subplots(figsize=(8, 6))
        data_melt = data.melt(id_vars=['date'])
        sns.lineplot(x="date", y="value", hue="variable", data=data_melt)
        if not (save_filename is None):
            self.save_fig(plt, self.get_save_file_path(save_filename))
        if show:
            plt.show()
        plt.close()

    def get_save_file_path(self, save_filename: str):
        """
        저장 파일 경로
        :param save_filename: 저장 파일명
        :return:
        """
        file_path = os.path.join(self.RESULTS_DIR, save_filename)
        DataUtils.create_dir(os.path.dirname(file_path))
        return file_path

    def draw_line_chart_scaled(self, data: pd.DataFrame):
        """
        선 그래프를 그린다.
        :param data: 데이터
        """
        scaler = MinMaxScaler()
        data_scaled = data
        data_value = data.iloc[:, 1:]
        data_scaled.iloc[:, 1:] = pd.DataFrame(scaler.fit_transform(data_value), columns=data_value.columns)
        self.draw_line_chart(data_scaled)

    def draw_line_charts(self, data: pd.DataFrame, dir_path: str = ''):
        """
        각 컬럼의 선 그래프를 저장한다.
        :param data: 데이터
        :param dir_path: 저장할 폴더 이름
        """
        dir_path = os.path.join("results/line_charts", dir_path)
        for col in data.columns:
            if col == 'date':
                continue
            self.draw_line_chart(data[['date', col]])
            self.save_fig(plt, os.path.join(dir_path, "line_chart_{}".format(col)))

    def draw_learning_curves(self, history, save_filename: str = "lstm_learning_curve.png", show: bool = False):
        """
        학습률 그래프를 그린다.
        :param history: 학습 히스토리
        :param save_filename: 저장 파일명
        :param show: 화면에 보녀줄지 여부
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history.history['loss'], label='Train', linewidth=3)
        ax.plot(history.history['val_loss'], label='Validation', linewidth=3)
        ax.set_title('Model loss', fontsize=16)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(loc='upper right')
        self.save_fig(plt, os.path.join(self.RESULTS_DIR, "learning", save_filename))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def save_fig(plot, file_path):
        DataUtils.create_dir(os.path.dirname(file_path))
        plot.savefig(file_path)
