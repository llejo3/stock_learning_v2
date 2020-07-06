from datetime import datetime, date

from dateutil import relativedelta

import logging_config as log
import pandas as pd


class DateUtils:
    """
    날짜 관련 메소드
    """

    # 기본 날짜 포멧
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)

    @staticmethod
    def add_years(d: date, add: int = 1) -> date:
        return d + relativedelta.relativedelta(years=add)

    @staticmethod
    def add_months(d: date, add: int = 1) -> date:
        return d + relativedelta.relativedelta(months=add)

    @staticmethod
    def add_days(d: date, add: int = 1):
        if add == 0:
            return d
        else:
            return d + relativedelta.relativedelta(days=add)

    @staticmethod
    def get_today_str(format_str: str = DATE_FORMAT) -> str:
        """
        오늘의 날짜를 가져온다.
        :param format_str: 날짜 포멧
        :return: 오늘의 날짜
        """
        return datetime.today().strftime(format_str)

    @staticmethod
    def to_str_date(d: date, format_str: str = DATE_FORMAT) -> str:
        """
        날짜를 문자형태로 변경한다.
        :param d: 날짜
        :param format_str: 날짜 포멧
        :return:
        """
        return d.strftime(format_str)

    @staticmethod
    def get_today() -> date:
        """
        오늘의 날짜를 가져온다.
        :return: 오늘의 날짜
        """
        return datetime.today().date()

    @staticmethod
    def to_date(date_str: str, date_format: str = DATE_FORMAT) -> date:
        """
        문자열을 데이트 형대로 변환한다.
        :param date_str: 날짜 문자열
        :param date_format: 날짜 포멧
        :return:
        """
        return datetime.strptime(date_str, date_format)

    @staticmethod
    def series_to_date(series: pd.Series):
        return series.dt.strftime(DateUtils.DATE_FORMAT).values[0]
