import os


class InvestConfig:
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    INVEST_PATH = os.path.join(ROOT_PATH, "results", "invest")

    @property
    def best_file_path(self):
        return os.path.join(self.INVEST_PATH, "best_result.txt")

    @property
    def searched_file_path(self):
        return os.path.join(self.INVEST_PATH, "search_result.txt")