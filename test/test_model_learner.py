from unittest import TestCase

from model_learner import ModelLearner
from stock_loader import StockLoader


class TestModelLearner(TestCase):

    def setUp(self):
        self.learner = ModelLearner()
        self.loader = StockLoader()

    def test_train_n_predict_by_lstm(self):
        corp_code = "016250"
        data = self.loader.get_stock_data(corp_code)
        self.learner.trains_n_predict_by_dnn(data, corp_code, pred_days=20, columns=["close"], model_type="lstm",
                                             lstm_neurons=[32])

    def test_search_grid_by_dnn(self):
        param_grid = {
            "time_steps": [5 * 2],
            "lstm_neurons": [[256, 128, 64]],
            "neurons": [[256, 128, 64]],
            "dropout": [0.1],
            "loss": ["mse"],
            "optimizer": ["adam"],
            "lstm_activation": ["tanh"],
            "activation": ["relu"],
            "batch_size": [64],
            "epochs": [500],
            "early_stopping_patience": [10],
            "model_type": ["mix"],
            "columns": [["close", "open", "high", "low", "volume"]]
        }
        corp_code = "006840"
        data = self.loader.get_stock_data(corp_code)
        self.learner.search_grid_by_dnn(data, corp_code=corp_code, pred_days=90, param_grid=param_grid)
