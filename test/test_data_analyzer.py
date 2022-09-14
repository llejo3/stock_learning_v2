import os
from unittest import TestCase

import pandas as pd
import tensorflow as tf

from data_analyzer import DataAnalyzer
from stock_investor import StockInvestor
from stock_loader import StockLoader
from utils.window_utils import WindowUtils


class TestDataAnalyzer(TestCase):

    def setUp(self):
        self.analyzer = DataAnalyzer()

    def test_predict_next(self):
        result = self.analyzer.predict_next("010140")
        print(result)

    def test_predict_period(self):
        result = self.analyzer.predict_period("006840", 60)
        print(result)

    def test_predicts_next(self):
        tf.config.set_visible_devices([], 'GPU')
        corps = ["KR모터스"]
        result = self.analyzer.predicts_next(corps, cnt_to_del=0, check_model=False)
        print(result)

    def test_predicts_next_for_best(self, update_stock=True):
        # tf.config.set_visible_devices([], 'GPU')
        bought_corp_names = ["유테크", "센트럴인사이트"]
        result = self.analyzer.predicts_next_for_best(update_stock=update_stock, cnt_to_del=0,
                                                      bought_corp_names=bought_corp_names, stored_model_only=True)
        print(result)

    def test_update_and_invest(self, update_stocks=True, init_result=True):
        if update_stocks:
            loader = StockLoader()
            loader.update_stocks()
        self.test_search_auto_investing_mock_all(init_result, 2)
        self.test_search_auto_investing_mock_all(False, 3)
        self.test_search_auto_investing_mock_all(False, 4)
        self.test_search_auto_investing_mock_all(False, 5)
        self.test_predicts_next_for_best(update_stock=False)
        WindowUtils.shutdown()

    def test_trains_all_and_invest(self, update_stocks=False, init_result=True):
        if update_stocks:
            loader = StockLoader()
            loader.update_stocks()
        self.analyzer.trains_all_only(model_expire_months=3, trying_cnt=1, pred_days=120, update_stock=False,
                                      cnt_to_del=0)
        self.test_search_auto_investing_mock_all(init_result, 2)
        self.test_search_auto_investing_mock_all(False, 3)
        self.test_search_auto_investing_mock_all(False, 4)
        self.test_search_auto_investing_mock_all(False, 5)
        self.test_predicts_next_for_best(update_stock=False)
        WindowUtils.shutdown()

    def test_search_auto_investing_mock_all(self, init_result=False, start_divisor=5):
        investor = StockInvestor()
        investor.search_auto_investing_mock_all(init_result=init_result, stored_model_only=True, update_stock=False,
                                                cnt_to_del=0, start_divisor=start_divisor, model_expire_months=6)

    def test_trains_all_only_cpu(self):
        tf.config.set_visible_devices([], 'GPU')
        self.analyzer.trains_all_only(model_expire_months=3, trying_cnt=3, pred_days=120, update_stock=True)

    def test_check_all_model_only(self):
        self.analyzer.check_all_model_only(drop=True, pred_days=120, update_stock=False)

    def test_tensorflow_gpu(self):
        print(tf.test.is_gpu_available())

    def test(self):
        # d = pd.read_csv("../results/invest/search_result.txt")
        print(int(0.993))
