import logging
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

import logging_config as log
from data_converter import DataConverter
from data_visualizer import DataVisualizer
from exceptions import ModelConfidenceError, ModelLearningError, ModelNotTrainError
from model_builder import ModelBuilder
from utils.data_utils import DataUtils
from utils.date_utils import DateUtils


class ModelLearner:
    """
    분석모델을 생성하고 학습한다.
    """

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    RESULTS_PATH = os.path.join(ROOT_PATH, "results")
    # 예측하는 값의 수
    PREDICT_PERIOD = 1
    # 예측하는 값의 구분
    PREDICT_FREQ = 'H'
    # 예측의 컬럼 이름
    COL_PRED = 'predict'

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)
        self.logger_level_info = self.logger.getEffectiveLevel() <= logging.INFO

    def search_grid_by_dnn(self, data: pd.DataFrame, param_grid: dict, corp_code: str = '', pred_days=30,
                           stored_model_only: bool = False):
        """
        DNN 모델에 여러가지 하이프파라미터를 시도한다.
        :param data: 데이터
        :param param_grid: 학습할 하이프파라미터
        :param corp_code: 주식회사 코드
        :param pred_days: 예측할 날들
        :param stored_model_only: 현재 저장되어 있는 모델만 하는지 여부
        :return:
        """
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in product(*values):
            params = dict(zip(keys, instance))
            try:
                self.trains_n_predict_by_dnn(data, corp_code, pred_days, stored_model_only, **params)
            except Exception as e:
                self.logger.info(e)

    def trains_n_predict_by_dnn(self, data: pd.DataFrame, corp_code: str = '', pred_days=0,
                                stored_model_only: bool = False, model_expire_months: int = 3,
                                **params):
        """
        LSTM 모델로  예측 값을 가져온다.
        :param data: 학습 데이터
        :param corp_code: 주식회사 코드
        :param pred_days: 예측하는 날수
        :param stored_model_only: 현재 저장되어 있는 모델만 하는지 여부
        :param model_expire_months: 모델의 유효한 달수
        :return: 예측 데이터
        """
        clear_session()
        best_model_path = self.get_best_model_path(corp_code)
        if os.path.exists(best_model_path):
            created_time = DataUtils.creation_time(best_model_path)
            if datetime.today() < DateUtils.add_months(created_time, model_expire_months):
                model, scalers = self.load_dnn_model(data, pred_days, corp_code, **params)
            else:
                model, scalers = self.trains_by_dnn(data, pred_days, corp_code, **params)
        elif stored_model_only is False:
            model, scalers = self.trains_by_dnn(data, pred_days, corp_code, **params)
        else:
            raise ModelNotTrainError(corp_code)
        if pred_days == 0:
            return self.predict_next_by_dnn(data, model, scalers, corp_code, **params)
        else:
            return self.predict_by_dnn(data, model, scalers, pred_days, corp_code, **params)

    def trains_by_dnn(self, data: pd.DataFrame, pred_days: int = 120, corp_code: str = '', check_model: bool = True,
                      **params):
        """
        DNN 모델을 여러번 돌려 최상의 모델을 가져온다.
        :param data: 데이터
        :param pred_days: 예측할 날 수
        :param corp_code: 주식회사 코드
        :param check_model: 모델을 체크하는지 여부
        :return:
        """
        self.logger.info("Started learning of a stock company with an item code of '{}' ...".format(corp_code))
        best_mse, best_model, best_result, scalers = self.trains_by_dnn_basic(data, pred_days, corp_code, **params)
        best_model, best_result = self.trains_by_dnn_exists_before(data, best_mse, best_model, best_result, pred_days,
                                                                   corp_code, **params)
        self.trains_by_dnn_checking_result(corp_code, best_result)
        if check_model is True:
            self.trains_by_dnn_checking_model(corp_code, best_model, best_result)
        self.logger.info("Ended learning of a stock company with an item code of '{}' ...".format(corp_code))
        return best_model, scalers

    def trains_by_dnn_basic(self, data: pd.DataFrame, pred_days: int = 120, corp_code: str = '', trying_cnt: int = 3,
                            **params):
        """
        DNN 모델을 여러번 돌려 최상의 모델을 가져오는 기본 메소드
        :param data:
        :param pred_days:
        :param corp_code:
        :param trying_cnt: 학습을 시도하는 횟수
        :param params:
        :return:
        """
        x_train, y_train, x_val, y_val, scalers = self.to_dnn_dataset(data, pred_days, **params)
        best_mse = None
        best_model = None
        best_result = None
        for i in range(trying_cnt):
            try:
                model = self.train_by_dnn(x_train, y_train, x_val, y_val, corp_code, **params)
            except Exception as e:
                self.logger.info(e)
                continue
            result, mse = self.predict_by_dnn(data, model, scalers, pred_days, corp_code, **params)
            if best_mse is None or best_mse > mse:
                best_mse = mse
                best_model = model
                best_result = result
        return best_mse, best_model, best_result, scalers

    def trains_by_dnn_exists_before(self, data: pd.DataFrame, best_mse, best_model, best_result, pred_days: int = 120,
                                    corp_code: str = '', **params):
        """
        DNN 모델을 여러번 돌려 최상의 모델을 가져온 후
        이전 모델과 비교한다.
        :param data:
        :param best_mse:
        :param best_model:
        :param best_result:
        :param pred_days:
        :param corp_code:
        :param params:
        :return:
        """
        best_model_path = self.get_best_model_path(corp_code)
        if os.path.exists(best_model_path):
            model, scalers = self.load_dnn_model(data, pred_days, corp_code, **params)
            result, mse = self.predict_by_dnn(data, model, scalers, pred_days, corp_code, **params)
            if best_mse is None or best_mse > mse:
                best_model = model
                best_result = result
        return best_model, best_result

    def trains_by_dnn_checking_result(self, corp_code, best_result):
        """
        DNN 모델을 여러번 돌려 최상의 모델을 가져온 후
        결과가 없을 경우 처리한다.
        :param corp_code:
        :param best_result:
        :return:
        """
        if best_result is None:
            DataUtils.remove_file(self.get_try_model_path(corp_code))
            error = ModelLearningError(corp_code)
            self.logger.info(error)
            raise error

    def trains_by_dnn_checking_model(self, corp_code, best_model, best_result):
        """
        DNN 모델을 여러번 돌려 최상의 모델을 가져온 후
        모델이 이상하지 않은지 체크한다.
        :param corp_code:
        :param best_model:
        :param best_result:
        :return:
        """
        if self.check_model(best_result) is False:
            DataUtils.remove_file(self.get_try_model_path(corp_code))
            DataUtils.remove_file(self.get_best_model_path(corp_code))
            # best_model.save_weights(self.get_unreliable_model_path(corp_code))
            error = ModelConfidenceError(corp_code)
            self.logger.info(error)
            raise error
        else:
            best_model_path = self.get_best_model_path(corp_code)
            best_model.save_weights(best_model_path)
            DataUtils.remove_file(self.get_try_model_path(corp_code))

    @staticmethod
    def check_model(best_result):
        cnt = np.sum(best_result.close > best_result.predict)
        ugly_cnt = np.sum(np.absolute(best_result.predict - best_result.close) / best_result.close > 0.3)
        data_len = len(best_result.index)
        if cnt == 0 or cnt == len(best_result.index) or ugly_cnt > data_len // 10:
            return False
        else:
            return True

    def train_by_dnn(self, x_train, y_train, x_val, y_val, corp_code='', batch_size=32, epochs=100,
                     early_stopping_patience=3, **params):
        """
        LSTM 모델로 학습을 한다.
        :param x_train: 학습 데이터 독립변수
        :param y_train: 학습 데이터 종속변수
        :param x_val: 검증 데이터 독립변수
        :param y_val: 검증 데이터 종속변수
        :param corp_code: 주식회사 코드
        :param epochs:
        :param batch_size:
        :param early_stopping_patience:
        :return:
        """
        es = EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience)
        best_model_path = self.get_try_model_path(corp_code)
        mc = ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True,
                             save_weights_only=True)
        model = self.get_dnn_models(x_train.shape, **params)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                            verbose=0, shuffle=True, callbacks=[mc, es])
        model.load_weights(best_model_path)
        if self.logger_level_info:
            save_filename = self.get_file_name("learning_curve", "png", corp_code)
            visualizer = DataVisualizer()
            visualizer.draw_learning_curves(history, save_filename)
        return model

    def to_dnn_dataset(self, data, pred_days=120, time_steps=32, columns=None, **params):
        """
        DNN을 위한 데이터로 변형한다.
        :param data:
        :param pred_days:
        :param params:
        :param time_steps:
        :param columns:
        :return:
        """
        if columns is None:
            columns = ["close"]
        size = pred_days + time_steps
        train = data[:-size]
        converter = DataConverter()
        x_train, y_train, scaler = converter.to_lstm_dataset(train, columns, time_steps, self.PREDICT_PERIOD)
        x_train, y_train, x_val, y_val = converter.split_train_validation(x_train, y_train)
        return x_train, y_train, x_val, y_val, scaler

    def load_dnn_model(self, data: pd.DataFrame, pred_days=120, corp_code='', time_steps=32, columns=("close",),
                       **params):
        """
        LSTM 모델로 학습을 한다.
        :param data: 학습 데이터
        :param pred_days: 예측 날의 수
        :param corp_code: 주식회사 코드
        :param time_steps:
        :param columns:
        :return:
        """
        y_size = self.PREDICT_PERIOD
        size = pred_days + time_steps
        train = data[:-size]
        converter = DataConverter()
        x_train, y_train, scalers = converter.to_lstm_dataset(train, columns, time_steps, y_size)
        model = self.load_model(x_train.shape, corp_code, **params)
        return model, scalers

    def load_model(self, x_shape, corp_code: str, **params):
        model = self.get_dnn_models(x_shape, **params)
        best_model_path = self.get_best_model_path(corp_code)
        model.load_weights(best_model_path)
        return model

    def get_best_model_path(self, corp_code):
        """
        가장 좋은 모델의 파일 이름
        :param corp_code:
        :return:
        """
        return os.path.join(self.RESULTS_PATH, "models", f"{corp_code}.h5")

    def get_unreliable_model_path(self, corp_code):
        """
        신뢰할 수 없는 모델을 저장하는 경로
        :param corp_code:
        :return:
        """
        return os.path.join(self.RESULTS_PATH, "models", "unreliable", f"{corp_code}.h5")

    def get_try_model_path(self, corp_code):
        """
        가장 좋은 모델을 시도하는 파일 이름
        :param corp_code:
        :return:
        """
        return os.path.join(self.RESULTS_PATH, "models", "try", f"{corp_code}.h5")

    @staticmethod
    def get_file_name(prefix, ext, corp_code):
        """
        학습 결과에 저장할 파일 이름
        :param prefix:
        :param ext:
        :param corp_code:
        :return:
        """
        return "{}(corp_code={}).{}".format(prefix, corp_code, ext)

    def get_file_path(self, dir_name, file_name):
        """
        파일 경로를 가져온다.
        파일 경로에 해당하는 폴더가 생성되지 않았으면 폴더도 생성한다.
        :param dir_name:
        :param file_name:
        :return:
        """
        path = os.path.join(self.RESULTS_PATH, dir_name, file_name)
        DataUtils.create_dir(os.path.dirname(path))
        return path

    @staticmethod
    def get_dnn_models(x_shape, y_size=PREDICT_PERIOD, model_type='mix', **model_params):
        """
        DNN 모델을 선택한다.
        :param x_shape:
        :param y_size:
        :param model_type:
        :return:
        """
        model_builder = ModelBuilder()
        model = None
        if model_type == "mix":
            model = model_builder.build_mix_model(x_shape, y_size, **model_params)
        elif model_type == "lstm":
            model = model_builder.build_lstm_model(x_shape, y_size, **model_params)
        elif model_type == "bidirectional":
            model = model_builder.build_lstm_model_bidirectional(x_shape, y_size, **model_params)
        elif model_type == "dense":
            model = model_builder.build_mlp_model(x_shape, y_size, **model_params)
        elif model_type == "cnn":
            model = model_builder.build_cnn_model(x_shape, y_size, **model_params)
        return model

    @staticmethod
    def get_dict_value(dic, key, default):
        """
        dict 의 데이터를 가져온다. default 값을 넣을 수 있다.
        :param dic:
        :param key:
        :param default: 기본 값
        :return:
        """
        if key in dic:
            value = dic[key]
        else:
            value = default
        return value

    @staticmethod
    def remove_dict_key(dic, *params):
        """
        dict 의 파라이터를 제거한다.
        :param dic:
        :param params:
        :return:
        """
        for key in params:
            if key in dic:
                del dic[key]

    def predict_by_dnn(self, data: pd.DataFrame, model=None, scalers: tuple = None, pred_days: int = 120, corp_code='',
                       **params):
        """
        LSTM 모델로 예측값을 가져온다.
        :param data: 학습 데이터
        :param model: 합습된 모델
        :param scalers: MinMaxScaler 들
        :param pred_days: 예측하는 날 수
        :param corp_code: 주식회사 코드
        :return:
        """
        y_size = self.PREDICT_PERIOD
        time_steps = self.get_dict_value(params, "time_steps", 32)
        columns = self.get_dict_value(params, "columns", ["close"])
        pred_size = pred_days
        size = pred_size + time_steps
        test = data.tail(size)
        converter = DataConverter()
        x_test, y_test, _ = converter.to_lstm_dataset(test, columns, time_steps, y_size, scalers=scalers)
        if model is None:
            model = self.load_model(x_test.shape, corp_code, **params)
        y_predict = model.predict(x_test)
        y_predict = scalers[1].inverse_transform(y_predict)
        y_pred_value = np.array([])
        remainder = pred_days % y_size
        merge_cnt = pred_days // y_size
        if remainder != 0:
            merge_cnt += 1
        for i in range(merge_cnt):
            y_pred_value = np.append(y_pred_value, y_predict[y_size * i])
        if remainder != 0:
            y_pred_value = y_pred_value[:-remainder]

        pred_data = test.tail(pred_size).reset_index(drop=True)
        mse = mean_squared_error(pred_data.close, y_pred_value)
        pred_data['predict'] = y_pred_value
        self.logger.info("DNN Model MSE({}, pred_days={}, corp_code={}): {}".format(params, pred_days, corp_code, mse))
        if self.logger_level_info:
            file_name = self.get_file_name("dnn_result_data", "pkl", corp_code)
            DataUtils.save_pickle(pred_data, self.get_file_path("pred_data", file_name))
            file_name = self.get_file_name("predict_by_dnn", "png", corp_code)
            visualizer = DataVisualizer()
            visualizer.draw_line_chart(pred_data[['date', 'close', 'predict']], os.path.join("pred_chart", file_name))
        return pred_data, mse

    def predict_next_by_dnn(self, data: pd.DataFrame, model=None, scalers: tuple = None, corp_code='', time_steps=32,
                            columns=("close",), **params):
        """
        LSTM 모델로 다은 날을 예측한다.
        :param data: 학습 데이터
        :param model: 합습된 모델
        :param scalers: MinMaxScaler 들
        :param corp_code: 주식회사 코드
        :param time_steps:
        :param columns:
        :return:
        """
        converter = DataConverter()
        x = converter.get_last_dataset(data, columns, time_steps, scalers)
        if model is None:
            model = self.load_model(x.shape, corp_code, **params)
        y_predict = model.predict(x)
        y_predict = scalers[1].inverse_transform(y_predict)
        return y_predict.item()
