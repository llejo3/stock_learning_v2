import re
from unittest import TestCase

import tensorflow as tf

from stock_investor import StockInvestor
from stock_loader import StockLoader


class TestStockInvestor(TestCase):

    def setUp(self):
        self.investor = StockInvestor()

    def test_invest_mock(self):
        params = {
            "buy_min_ratio": 13,
            "sell_min_ratio": 51,
            "take_profit_ratio": 196,
            "stop_loss_ratio": 11,
            "take_profit_pred_ratio": 10
        }
        mock, index, dates = self.investor.invest_mock("006920", update_stock=False, **params)
        print(mock, index, dates)

    def test_invests_mock_all(self):
        result = self.investor.invests_mock_all(stored_model_only=True, model_expire_months=1, trying_cnt=1)
        print(result.head())

    def test_mean_investing_mock_all(self):
        params = {
            "buy_min_ratio": 28,
            "take_profit_1_ratio": 21,
            "take_profit_2_ratio": 22,
            "take_profit_3_ratio": 23,
            "stop_loss_ratio": 10
        }
        tf.config.set_visible_devices([], 'GPU')
        mean, _ = self.investor.mean_investing_mock_all(params, stored_model_only=True, update_stock=False,
                                                        cnt_to_del=1)
        print(mean)

    def test_search_grid_investing_mock_all(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                "buy_min_ratio": [28],
                "take_profit_1_ratio": [22],
                "take_profit_2_ratio": [23],
                "stop_loss_ratio": [10]
            }
        self.investor.search_grid_investing_mock_all(param_grid, stored_model_only=True, update_stock=False,
                                                     cnt_to_del=0, model_expire_months=6)

    def test_search_grid_investing_mock_all_cpu(self):
        param_grid = {
            "buy_min_ratio": [28],
            "sell_min_ratio": [11],
            "take_profit_ratio": [43],
            "stop_loss_ratio": [10]
        }
        tf.config.set_visible_devices([], 'GPU')
        self.test_search_grid_investing_mock_all(param_grid)

    def test_search_grid_investing_mock_all_cpu2(self):
        param_grid = {
            "buy_min_ratio": [13],
            "sell_min_ratio": [44],
            "take_profit_ratio": [196],
            "stop_loss_ratio": [7, 9, 6, 10]
        }
        tf.config.set_visible_devices([], 'GPU')
        self.test_search_grid_investing_mock_all(param_grid)

    def test_search_random_investing_mock_all(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                "buy_min_ratio": [17, 20, 23],
                "sell_min_ratio": [27, 30, 33],
                "take_profit_ratio": [22, 25, 27],
                "stop_loss_ratio": [22, 25, 27]
            }
        self.investor.search_random_investing_mock_all(param_grid, random_cnt=100, stored_model_only=True,
                                                       update_stock=False)

    def test_search_random_investing_mock_all_cpu(self):
        tf.config.set_visible_devices([], 'GPU')
        self.test_search_random_investing_mock_all()

    def test_search_random_investing_mock_all_cpu2(self):
        tf.config.set_visible_devices([], 'GPU')
        self.test_search_random_investing_mock_all()

    def test_search_random_investing_mock_all_cpu3(self):
        tf.config.set_visible_devices([], 'GPU')
        self.test_search_random_investing_mock_all()

    def test_invests_best_mock(self):
        result = self.investor.invests_best_mock(stored_model_only=True, update_stock=False, best_cnt=20)
        print(result)

    def test_get_random_products(self):
        param_grid = {
            "buy_min_ratio": range(0, 31, 1),
            "sell_min_ratio": range(0, 31, 1),
            "take_profit_ratio": range(0, 31, 1),
            "stop_loss_ratio": range(0, 31, 1),
        }
        products = self.investor.get_random_products(param_grid)
        print(products)

    def test_search_auto_investing_mock_all(self):
        # loader = StockLoader()
        # loader.update_stocks()
        tf.config.set_visible_devices([], 'GPU')
        self.investor.search_auto_investing_mock_all(init_result=False, stored_model_only=True, update_stock=False,
                                                     cnt_to_del=0, start_divisor=2, model_expire_months=6)

    def test_search_auto_investing_mock_all2(self):
        tf.config.set_visible_devices([], 'GPU')
        self.investor.search_auto_investing_mock_all(init_result=False, stored_model_only=True, update_stock=False,
                                                     cnt_to_del=0, start_divisor=3, model_expire_months=6)

    def test_search_auto_investing_mock_all3(self):
        tf.config.set_visible_devices([], 'GPU')
        self.investor.search_auto_investing_mock_all(init_result=False, stored_model_only=True, update_stock=False,
                                                     cnt_to_del=0, start_divisor=4, model_expire_months=6)

    def test_search_auto_investing_mock_all4(self):
        # tf.config.set_visible_devices([], 'GPU')
        self.investor.search_auto_investing_mock_all(init_result=False, stored_model_only=True, update_stock=False,
                                                     cnt_to_del=0, start_divisor=5, model_expire_months=6)

    def test_search_auto_samples_investing_mock_all(self):
        tf.config.set_visible_devices([], 'GPU')
        self.investor.search_auto_samples_investing_mock_all(cnt_to_del=0)

    def test_add_stock_unit(self):
        print(self.investor.add_stock_unit(2000, 2))
        print(self.investor.add_stock_unit(20000))
        print(self.investor.add_stock_unit(200000))

    def test_subtract_stock_unit(self):
        print(self.investor.subtract_stock_unit(2000, 2))
        print(self.investor.subtract_stock_unit(20034))
        print(self.investor.subtract_stock_unit(200001))