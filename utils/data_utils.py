import os
import platform
from datetime import datetime

import pandas as pd

import logging_config as log


class DataUtils:
    """
    데이터 처리 관련 메소드
    """

    @staticmethod
    def save_pickle(df_data: pd.DataFrame, file_path: str):
        """
        Pickle 형태의 데이터로 저장한다.
        :param df_data:
        :param file_path:
        :return:
        """
        DataUtils.create_dir(os.path.dirname(file_path))
        df_data.to_pickle(file_path)

    @staticmethod
    def save_csv(df_data: pd.DataFrame, file_path: str):
        """
        csv 형태의 데이터로 저장한다.
        :param df_data:
        :param file_path:
        :return:
        """
        DataUtils.create_dir(os.path.dirname(file_path))
        df_data.to_csv(file_path, index=False, encoding="utf-8")

    @staticmethod
    def update_csv(data: pd.DataFrame, file_path: str, sort_by=None, ascending=True):
        """
        csv 형태의 데이터로 저장한다.
        :param ascending:
        :param sort_by:
        :param data:
        :param file_path:
        :return:
        """
        if os.path.exists(file_path):
            saved_data = pd.read_csv(file_path, encoding="utf-8")
            data = pd.concat([saved_data, data], ignore_index=True)
        if not (sort_by is None):
            data = data.sort_values(by=sort_by, ascending=ascending)
        DataUtils.save_csv(data, file_path)
        return data

    @staticmethod
    def create_dir(dir_path: str):
        """
        디렉토리를 생성한다.
        :param dir_path:
        :return:
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def creation_time(path_to_file):
        """
        Try to get the date that a file was created, falling back to when it was
        last modified if that isn't possible.
        See http://stackoverflow.com/a/39501288/1709587 for explanation.
        """
        if platform.system() == 'Windows':
            time = os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                time = stat.st_birthtime
            except AttributeError:
                # We're probably on Linux. No easy way to get creation dates here,
                # so we'll settle for when its content was last modified.
                time = stat.st_mtime
        return datetime.fromtimestamp(time)

    @staticmethod
    def remove_file(file_path: str):
        """
        파일이 존재하면 삭제한다.
        :param file_path: 파일 경로
        :return:
        """
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger = log.get_logger("DataUtils")
                logger.error(e)

    @staticmethod
    def put_first(n: int, df: pd.DataFrame):
        df = df.reindex([n] + list(range(0, n)) + list(range(n+1, df.index[-1] + 1))).reset_index(drop=True)
        return df
