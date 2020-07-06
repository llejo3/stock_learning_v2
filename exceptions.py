class ModelConfidenceError(Exception):
    """
    모델을 신뢰할 수 없을 경우의 에러
    """

    def __init__(self, corp_code=''):
        self.corp_code = corp_code

    def __str__(self):
        return "The model of the stock company with the item code '{}' can't be trusted.".format(self.corp_code)


class ModelLearningError(Exception):
    """
    모델을 학습하는 중 에러 발생
    """

    def __init__(self, corp_code=''):
        self.corp_code = corp_code

    def __str__(self):
        return "An error occurred while learning a model of a stock company '{}'".format(self.corp_code)


class ModelNotTrainError(Exception):
    """
    모델을 학습하는 중 에러 발생
    """

    def __init__(self, corp_code=''):
        self.corp_code = corp_code

    def __str__(self):
        return "There are no saved learning models. '{}'".format(self.corp_code)
