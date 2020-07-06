import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import logging_config as log


class DataConverter:
    """
    데이터를 변환한다.
    """

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)

    @staticmethod
    def get_category_data(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        해당 카테고리 데이터만 가져온다.
        :param data: 데이터
        :param prefix:
        :return:
        """
        column_basic = ['date', 'total_cnt', 'volume']
        column_cate = [column for column in data.columns if column.startswith(prefix)]
        column_basic.extend(column_cate)
        return data[column_basic]

    @staticmethod
    def del_zero_column(data: pd.DataFrame):
        """
        모든 데이터가 zero 인 컬럼을 제거한다.
        :param data: 데이터
        """
        desc = data.describe()
        col_zero = desc.columns[desc.loc['max',] == 0].values
        if len(col_zero) > 0:
            data.drop(columns=col_zero, inplace=True)

    @staticmethod
    def to_date_column(data: pd.DataFrame):
        """
        날짜 컬럼을 하나의 컬럼으로 변환한다.
        :param data: 데이터
        """
        data['date'] = pd.to_datetime(data.dt, format='%Y-%m-%d') + pd.to_timedelta(data.hh, unit='h')
        data.drop(columns=['dt', 'hh'], inplace=True)

    @staticmethod
    def del_intermittent_column(data: pd.DataFrame):
        """
        간헐적인 데이터를 제거한다.
        :param data: 데이터
        """
        intermittent_columns = ['source_custom', 'source_google_plus']
        data.drop(columns=intermittent_columns, inplace=True)

    @staticmethod
    def to_fb_prophet_type(data: pd.DataFrame, date_col_name='ds', y_col_name='y'):
        """
        Facebook Prophet 형태의 데이터로 변환한다.
        :param data: 데이터
        :param date_col_name: 날짜 형태의 컬럼명
        :param y_col_name: 종속변수의 걸럼명
        :return: Facebook Prophet 형태의 데이터
        """
        return data[[date_col_name, y_col_name]].rename(columns={date_col_name: 'ds', y_col_name: 'y'})

    @staticmethod
    def to_lstm_dataset(data, columns, time_steps=120, y_size=1, scalers: tuple = None) -> tuple:
        """
        LSTM 형태의 데이터로 변형한다.
        :param data: 데이터
        :param columns : 데이터 커럼들
        :param time_steps: 순서열의 길이
        :param y_size: 결과의 길이
        :param scalers: MinMaxScaler 들
        :return: LSTM 을 위한 데이터
        """
        if scalers is None:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaler = x_scaler.fit(data[columns])
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler = y_scaler.fit(data[['close']])
            scalers = (x_scaler, y_scaler)
        x_normalized = scalers[0].transform(data[columns])
        y_normalized = scalers[1].transform(data[['close']])
        xs, ys = [], []
        for i in range(len(x_normalized) - time_steps - y_size + 1):
            x = x_normalized[i:i + time_steps]
            xs.append(x)
            y = y_normalized[i + time_steps: i + time_steps + y_size][:, 0]
            ys.append(y)
        xs = np.array(xs).astype(np.float32)
        return xs, np.array(ys), scalers

    @staticmethod
    def get_last_dataset(data, columns, time_steps: int, scalers: tuple) -> np.ndarray:
        """
        LSTM 형태의 데이터로 변형한다.
        :param data: 데이터
        :param columns : 데이터 커럼들
        :param time_steps: 순서열의 길이
        :param scalers: MinMaxScaler 들
        :return: LSTM 을 위한 데이터
        """
        x_normalized = scalers[0].transform(data[columns])
        x_len = len(x_normalized)
        x = x_normalized[x_len - time_steps:]
        return np.array([x]).astype(np.float32)

    @staticmethod
    def split_train_validation(x, y) -> tuple:
        """
        학습데이터와 검증데이터를 분리한다.
        :param x: X 데이터
        :param y: y 데이터
        :return: 분리된 데이터
        """
        x_train, x_val, y_train, y_val = train_test_split(np.array(x), y, test_size=0.2, shuffle=True, random_state=42)
        return x_train, y_train, x_val, y_val

    @staticmethod
    def zfill_stock_code(data: pd.DataFrame, col_name: str = '종목코드'):
        """
        종목 코드를 6자리로 채운다.
        :param data:
        :param col_name:
        :return:
        """
        data.loc[:, col_name] = data[col_name].astype(str).str.zfill(6)
