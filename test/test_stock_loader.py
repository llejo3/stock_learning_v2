from unittest import TestCase

from stock_loader import StockLoader


class TestStockLoader(TestCase):

    def setUp(self):
        self.loader = StockLoader()

    def test_get_stock_data(self):
        data = self.loader.get_stock_data("016250")
        print(data.head())

    def test_crawl_stock_daum(self):
        data = self.loader.crawl_stock_daum("016250")
        print(data.head())

    def test_crawl_stock_naver(self):
        data = self.loader.crawl_stock_naver("016250")
        print(data.head())

    def test_update_stocks(self):
        self.loader.update_stocks()
