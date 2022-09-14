import os
from pathlib import Path


class InvestConfig:
    ROOT_PATH = Path(os.path.abspath(__file__)).parent
    INVEST_PATH = ROOT_PATH / "results" / "invest"

    def __init__(self):
        self.INVEST_PATH.mkdir(exist_ok=True, parents=True)

    @property
    def best_file_path(self):
        return self.INVEST_PATH / "best_result.txt"

    @property
    def searched_file_path(self):
        return self.INVEST_PATH / "search_result.txt"
