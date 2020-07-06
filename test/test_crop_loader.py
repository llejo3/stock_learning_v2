from unittest import TestCase

from corp_loader import CorpLoader


class TestCropLoader(TestCase):

    def setUp(self):
        self.loader = CorpLoader()

    def test_get_corps_all(self):
        data = self.loader.get_corps_all()
        print(data.head())
        print(data.dtypes)

    def test_get_corps_master(self):
        data = self.loader.crawl_corps_master()
        print(data.head())

    def test_get_corps_market_cap(self):
        data = self.loader.get_corps_market_cap()
        print(data.head())

    def test_get_crops_with_market_cap(self):
        data = self.loader.get_crops_with_market_cap()
        print(data.head())

    def test_get_eval_crops(self):
        data = self.loader.get_eval_crops()
        print(data)
        print(data.query(f"회사명=='일신석재'"))