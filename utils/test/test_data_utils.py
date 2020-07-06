from unittest import TestCase

from utils.data_utils import DataUtils


class TestDataUtils(TestCase):

    def test_creation_time(self):
        time = DataUtils.creation_time("../../results/models/001250.h5")
        print(time)
