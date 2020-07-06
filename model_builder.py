import logging

from tensorflow.keras.layers import Conv1D, Dense, Input, LSTM, Dropout, concatenate, Bidirectional, Flatten
from tensorflow.keras.models import Sequential, Model

import logging_config as log


class ModelBuilder:

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)

    def print_model(self, model):
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            model.summary()
        names = [layer.get_config()["name"] for layer in model.layers]
        self.logger.debug("Layers : {}".format(names))

    def build_lstm_model(self, input_shape, output_size, lstm_neurons: list, dropout=0.10, loss="mse",
                         optimizer="adam", lstm_activation="tanh", **params):
        """
        LSTM 모델
        :param input_shape:
        :param output_size:
        :param lstm_neurons:
        :param dropout:
        :param loss:
        :param lstm_activation:
        :param optimizer:
        :return:
        """
        model = Sequential()
        for i, n in enumerate(lstm_neurons):
            p = {}
            if i == 0:
                p["input_shape"] = (input_shape[1], input_shape[2])
            if i != len(lstm_neurons) - 1:
                p["return_sequences"] = True
            model.add(LSTM(n, dropout=dropout, activation=lstm_activation, **p))
        model.add(Dense(units=output_size))
        model.compile(loss=loss, optimizer=optimizer)
        self.print_model(model)
        return model

    def build_lstm_model_bidirectional(self, input_shape, output_size, lstm_neurons: list, dropout=0.10, loss="mse",
                                       optimizer="adam", lstm_activation="tanh", **params):
        """
        Bidirectional LSTM 모델
        :param input_shape:
        :param output_size:
        :param lstm_neurons:
        :param dropout:
        :param loss:
        :param optimizer:
        :param lstm_activation:
        :return:
        """
        model = Sequential()

        for i, n in enumerate(lstm_neurons):
            params = {}
            if i == 0:
                params["input_shape"] = (input_shape[1], input_shape[2])
            if i != len(lstm_neurons) - 1:
                params["return_sequences"] = True
            model.add(Bidirectional(LSTM(n, dropout=dropout, activation=lstm_activation), **params))
        model.add(Dense(units=output_size))
        model.compile(loss=loss, optimizer=optimizer)
        self.print_model(model)
        return model

    def build_mlp_model(self, input_shape, output_size, neurons: list, dropout=0.30, loss="mse", optimizer="adam",
                        activation="relu", **params):
        """
        MLP 모델
        :param input_shape:
        :param output_size:
        :param neurons:
        :param dropout:
        :param loss:
        :param optimizer:
        :param activation:
        :return:
        """
        model = Sequential()
        for i, n in enumerate(neurons):
            params = {}
            if i == 0:
                params["input_shape"] = (input_shape[1], input_shape[2])
            model.add(Dense(n, activation=activation, **params))
            model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(units=output_size))
        model.compile(loss=loss, optimizer=optimizer)
        self.print_model(model)
        return model

    def build_cnn_model(self, input_shape, output_size, neurons: list, dropout=0.30, loss="mse", optimizer="adam",
                        activation="relu", **params):
        """
        CNN 모델
        :param input_shape:
        :param output_size:
        :param neurons:
        :param dropout:
        :param loss:
        :param optimizer:
        :param activation:
        :return:
        """
        model = Sequential()
        for i, n in enumerate(neurons):
            params = {}
            if i == 0:
                params["input_shape"] = (input_shape[1], input_shape[2])
            model.add(Conv1D(n, 3, activation=activation, **params))
            model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(units=output_size))
        model.compile(loss=loss, optimizer=optimizer)
        self.print_model(model)
        return model

    def build_mix_model(self, input_shape, output_size, lstm_neurons: list, neurons: list = None, dropout=0.10,
                        loss="mse", optimizer="adam", lstm_activation="tanh", activation="relu", **params):
        """
        LSTM, CNN, MLP 모델이 함께 들어간 모델
        :param input_shape:
        :param output_size:
        :param lstm_neurons:
        :param neurons:
        :param dropout:
        :param loss:
        :param optimizer:
        :param lstm_activation:
        :param activation:
        :return:
        """
        input_model = Input(shape=(input_shape[1], input_shape[2]))
        lstm = self.get_mix_lstm_model(input_model, lstm_neurons, input_shape, dropout, lstm_activation, output_size)
        cnn = self.get_mix_cnn_model(input_model, neurons, input_shape, dropout, activation, output_size)
        mlp = self.get_mix_mlp_model(input_model, neurons, input_shape, dropout, activation, output_size)
        m = concatenate([lstm, cnn, mlp])
        m = Dense(units=output_size)(m)
        model = Model(input_model, m)
        model.compile(optimizer=optimizer, loss=loss)
        self.print_model(model)
        return model

    @staticmethod
    def get_mix_lstm_model(input_model, lstm_neurons, input_shape, dropout, lstm_activation, output_size):
        lstm = input_model
        for i, n in enumerate(lstm_neurons):
            p = {}
            if i == 0:
                p["input_shape"] = (input_shape[1], input_shape[2])
            if i != len(lstm_neurons) - 1:
                p["return_sequences"] = True
            lstm = LSTM(n, dropout=dropout, activation=lstm_activation, **p)(lstm)
        lstm = Flatten()(lstm)
        lstm = Dense(units=output_size)(lstm)
        return lstm

    @staticmethod
    def get_mix_cnn_model(input_model, neurons, input_shape, dropout, activation, output_size):
        cnn = input_model
        for i, n in enumerate(neurons):
            p = {}
            if i == 0:
                p["input_shape"] = (input_shape[1], input_shape[2])
            cnn = Conv1D(n, 3, activation=activation, **p)(cnn)
            cnn = Dropout(dropout)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(units=output_size)(cnn)
        return cnn

    @staticmethod
    def get_mix_mlp_model(input_model, neurons, input_shape, dropout, activation, output_size):
        mlp = input_model
        for i, n in enumerate(neurons):
            p = {}
            if i == 0:
                p["input_shape"] = (input_shape[1], input_shape[2])
            mlp = Dense(n, activation=activation, **p)(mlp)
            mlp = Dropout(dropout)(mlp)
        mlp = Flatten()(mlp)
        mlp = Dense(units=output_size)(mlp)
        return mlp
