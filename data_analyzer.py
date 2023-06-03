import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from keras.backend import clear_session
from tqdm import tqdm

import logging_config as log
from config import InvestConfig
from corp_loader import CorpLoader
from model_learner import ModelLearner
from stock_loader import StockLoader
from utils.data_utils import DataUtils
from utils.date_utils import DateUtils


class DataAnalyzer:
    """
    데이터 분석을 실행한다.
    """
    ROOT_PATH = Path(os.path.abspath(__file__)).parent
    PREDICT_PATH = ROOT_PATH / "results" / "analyze"

    MODEL_PARAMS = {
        "time_steps": 5 * 2,
        "lstm_neurons": [256, 128, 64],
        "neurons": [256, 128, 64],
        "dropout": 0.1,
        "loss": "mse",
        "optimizer": "adam",
        "lstm_activation": "tanh",
        "activation": "relu",
        "batch_size": 64,
        "epochs": 200,
        "early_stopping_patience": 5,
        "model_type": "mix",
        "columns": ["close", "open", "high", "low", "volume"]
    }

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)
        self.PREDICT_PATH.mkdir(parents=True, exist_ok=True)

    def predicts_next_for_best(self, best_cnt=None, bought_corp_names: list = None, **params):
        invest_cfg = InvestConfig()
        invest_data = pd.read_csv(invest_cfg.searched_file_path)
        invest_first = invest_data.iloc[0].to_dict()
        print(invest_first)
        params.update(invest_first)
        best_data = pd.read_csv(invest_cfg.best_file_path)
        if best_cnt is None:
            # line = max(invest_first['mean_value'], 10000000)
            line = 10000000
            best_data = best_data.query(f"predict>{line}")
        else:
            best_data = best_data[:best_cnt]
        best_data.loc[:, 'code'] = best_data['code'].astype(str).str.zfill(6)
        if bought_corp_names is not None:
            best_data = self.update_bought_corps(bought_corp_names, best_data)
        return self.predicts_next(best_data, **params)

    @staticmethod
    def update_bought_corps(bought_corp_names, best_data):
        corps = None
        for bought_corp_name in bought_corp_names:
            bought_corp_index = best_data.index[best_data['name'] == bought_corp_name]
            if len(bought_corp_index) > 0:
                best_data.loc[best_data['name'] == bought_corp_name, 'name'] = f"{bought_corp_name}(Bought)"
                # index = bought_corp_index[0]
                # best_data = DataUtils.put_first(index, best_data)
            else:
                if corps is None:
                    corp = CorpLoader()
                    corps = corp.get_corps_all()
                corp = corps.query(f"회사명=='{bought_corp_name}'").reset_index(drop=True)
                if len(corp.index) > 0:
                    corp_name = f"{corp.loc[0, '회사명']}(Bought New)"
                    corp = pd.DataFrame([{'name': corp_name, 'code': corp.loc[0, '종목코드']}])
                    best_data = best_data.append(corp, ignore_index=True)
        return best_data

    def predicts_next(self, corps: pd.DataFrame, buy_min_ratio=25, **params):
        next_data = []
        for row in tqdm(corps.itertuples(), total=len(corps.index), desc="Predicts next"):
            corp_code = getattr(row, 'code')
            corp_name = getattr(row, 'name')
            index = getattr(row, 'Index')
            try:
                next_row = self.predict_next(corp_code, **params)
                row = [index + 1, corp_code, corp_name]
                row.extend(self.get_values(next_row, "date", "close", "next_close", "ratio", "cnt"))
                next_data.append(row)
            except Exception as e:
                self.logger.error(f"Corp Code: {corp_code}")
                self.logger.error(DataUtils.get_error_message(e))
        next_data = pd.DataFrame(next_data,
                                 columns=["rank", "code", "name", "date", "close", "next_close", "ratio", "cnt"])
        tops = next_data.loc[next_data['ratio'] >= buy_min_ratio].sort_values(by='ratio', ascending=False)
        bottoms = next_data.loc[next_data['ratio'] < buy_min_ratio].sort_values(by='ratio', ascending=False)
        next_data = pd.concat([tops, pd.DataFrame([{"name": "___"}])], ignore_index=True)
        next_data = pd.concat([next_data, bottoms], ignore_index=True)
        # next_data["remain_days"] = self.get_remain_days(max_stay_days)
        # next_data = next_data.sort_values(by='ratio', ascending=False)
        next_date = next_data.tail(1).date.dt.strftime(DateUtils.DATE_FORMAT).values[0]
        file_name = f"next_{next_date}.txt"
        DataUtils.save_csv(next_data, os.path.join(self.PREDICT_PATH, file_name))
        return next_data

    @staticmethod
    def get_remain_days(max_stay_days=20):
        day = 90 - ((max_stay_days // 5)*7 + max_stay_days % 5)
        return day

    @staticmethod
    def get_values(data: pd.DataFrame, *columns: str):
        result = []
        for column in columns:
            result.append(data[column].values[0])
        return result

    def predict_next(self, corp_code: str, cnt_to_del: int = 0, update_stock=True, **params):
        """
        다음 날을 예측한다.
        :param corp_code: 예측하는 주식회사 코드
        :param cnt_to_del: 마지막 데이터 삭제하는 개수
        :param update_stock: 주식데이터를 업데이트 하는지 여부
        :return:
        """
        stock = StockLoader()
        learner = ModelLearner()
        data = stock.get_stock_data(corp_code, cnt_to_del=cnt_to_del, update_stock=update_stock)
        params.update(self.MODEL_PARAMS)
        next_close = learner.trains_n_predict_by_dnn(data, corp_code, **params)
        next_data = data.tail(1)[['date', 'close']].reset_index(drop=True)
        next_data['next_close'] = next_close
        close = next_data.close.values[0]
        next_data['ratio'] = np.round((next_close / close) * 100 - 100, 2)
        next_data['cnt'] = np.round(1000000 / next_data['close'])
        next_data.close = next_data.close.astype(int)
        next_data.next_close = next_data.next_close.astype(int)
        self.logger.debug(next_data.to_dict("records"))
        return next_data

    # @DataUtils.clock
    def predict_period(self, corp_code: str, cnt_to_del=0, update_stock=True, **params) -> pd.DataFrame:
        stock = StockLoader()
        learner = ModelLearner()
        data = stock.get_stock_data(corp_code, cnt_to_del, update_stock)
        params.update(self.MODEL_PARAMS)
        predicts, _ = learner.trains_n_predict_by_dnn(data, corp_code, **params)
        return predicts

    def trains_all_only(self, start=0, end: int = None, **params):
        corps = self.get_corps_for_train(start, end)
        for row in tqdm(corps.itertuples(index=False), total=len(corps.index), desc="Train models"):
            corp_code = getattr(row, "종목코드")
            try:
                self.train_only(corp_code, **params)
            except Exception as e:
                self.logger.error(DataUtils.get_error_message(e))

    @staticmethod
    def get_corps_for_train(start=0, end: int = None) -> pd.DataFrame:
        corp = CorpLoader()
        corps = corp.get_crops_confidence()
        if start != 0 or not (end is None):
            if end is None:
                corps = corps[start:]
            else:
                corps = corps[start:end]
        return corps

    def check_all_model_only(self, pred_days: int = 60, drop=False, update_stock=True):
        corp = CorpLoader()
        corps = corp.get_crops_confidence()
        for row in tqdm(corps.itertuples(index=False), total=len(corps.index), desc="Model checking"):
            corp_code = getattr(row, "종목코드")
            try:
                self.check_model_only(corp_code, pred_days, drop, update_stock)
            except Exception as e:
                self.logger.error(DataUtils.get_error_message(e))

    def check_model_only(self, corp_code, pred_days: int = 60, drop=False, update_stock=True):
        learner = ModelLearner()
        best_model_path = learner.get_best_model_path(corp_code)
        reliable = False
        if os.path.exists(best_model_path):
            stock = StockLoader()
            data = stock.get_stock_data(corp_code, update_stock=update_stock)
            clear_session()
            model, scalers = learner.load_dnn_model(data, pred_days, corp_code, **self.MODEL_PARAMS)
            result, _ = learner.predict_by_dnn(data, model, scalers, pred_days, corp_code, **self.MODEL_PARAMS)
            if learner.check_model(result) is False:
                if drop is True:
                    DataUtils.remove_file(best_model_path)
                print("Unreliable model code '{}'".format(corp_code))
            else:
                reliable = True
        return reliable

    def train_only(self, corp_code, pred_days=60, cnt_to_del=0, model_expire_months=3, update_stock=True, **params):
        learner = ModelLearner()
        best_model_path = learner.get_best_model_path(corp_code)
        need_train = False
        if os.path.exists(best_model_path):
            created_time = DataUtils.creation_time(best_model_path)
            if datetime.today() >= DateUtils.add_months(created_time, model_expire_months):
                DataUtils.remove_file(best_model_path)
                need_train = True
            else:
                if not self.check_model_only(corp_code, pred_days, True, update_stock):
                    need_train = True
        else:
            need_train = True
        if need_train:
            self.train_model(corp_code, pred_days, cnt_to_del, update_stock, **params)

    def train_model(self, corp_code, pred_days=60, cnt_to_del=0, update_stock=True, **params):
        stock = StockLoader()
        data = stock.get_stock_data(corp_code, cnt_to_del, update_stock=update_stock)
        clear_session()
        params.update(self.MODEL_PARAMS)
        learner = ModelLearner()
        learner.trains_by_dnn(data, pred_days, corp_code, **params)
