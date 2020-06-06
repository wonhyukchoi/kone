import json
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable, Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import Input, Embedding, Flatten, Dense


class PredictorABC:
    """
    Abstract Base Class for all prediction modules.
    All prediction modules must follow this syntax,
    and should (ideally) implement all methods.
    """

    def __init__(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, *args, **kwargs):
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError


class Kone(PredictorABC):
    def __init__(self, window_size: int):
        super().__init__()
        self._window_size = window_size
        self._x_index = None
        self._y_index = None
        self._model = None
        self._train_history = None

    def train(self, x: Sequence, y: Sequence,
              padding_symbol='<p>',
              epochs=30, batch_size=1 << 14,
              embedding_dim=300, num_neurons=128,
              optimizer='rmsprop', verbose=1) -> None:

        flat_x, flat_y = "".join(x), "".join(y)
        self._x_index = IntIndex(vocabulary=flat_x)
        self._x_index.add_vocab(padding_symbol)
        self._y_index = IntIndex(vocabulary=flat_y)

        flat_x_dict = self._transform_x(text_list=x, padding_symbol=padding_symbol)
        matrix_x = flat_x_dict['matrix']
        matrix_y = self._transform_y(flat_y)

        x_train, x_test, y_train, y_test = train_test_split(matrix_x, matrix_y)

        self._build_model(embedding_dim=embedding_dim,
                          num_neurons=num_neurons)
        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        self._train_history = self._model.fit(x_train, y_train,
                                              batch_size=batch_size,
                                              validation_data=[x_test, y_test],
                                              epochs=epochs,
                                              verbose=verbose)

    def predict(self, text_list: Iterable) -> list:
        if not self._model:
            raise ReferenceError("Please train or load a model.")

        x_dict = self._transform_x(text_list)
        flat_x_chars = x_dict['flat_x']
        text_lengths = x_dict['text_lengths']
        windowed_and_numericized = x_dict['matrix']

        y_hat_matrix = self._model.predict(windowed_and_numericized)
        y_hat = np.argmax(y_hat_matrix, axis=1)
        noun_list = self._extract_nouns(char_list=flat_x_chars,
                                        predictions=y_hat,
                                        text_lengths=text_lengths)

        return noun_list

    def save_model(self,
                   index_name="index.json",
                   weight_name="weights.h5"):
        self._x_index.to_json(file_name=index_name)
        self._model.save(weight_name)

    def load_index(self, index_path):
        self._x_index = IntIndex.from_json(index_path)

    def load_weights(self, model_path):
        self._model = load_model(model_path)

    def load_model(self, index_path, model_path):
        """ Method to retain consistency with API syntax. """
        self.load_index(index_path=index_path)
        self.load_weights(model_path=model_path)

    # TODO: test this works
    def _transform_x(self, text_list: Iterable,
                     padding_symbol='<p>') -> dict:
        flat_x, text_lengths, matrix_list = "", [], []
        padding = padding_symbol * self._window_size
        seq_len = len(padding) * 2 + 1

        for text in text_list:
            flat_x += text
            text_lengths.append(len(text))

            padded_text = padding + text + padding
            padded_as_num = [self._x_index.to_int(char) for char in padded_text]

            numericized_padded = []
            for i in range(len(text)):
                numericized_padded += padded_as_num[i: i + seq_len]
            matrix_list.append(numericized_padded)

        matrix = np.array(matrix_list)
        return dict(flat_x=flat_x, text_lengths=text_lengths,
                    matrix=matrix)

    # TODO: test this works
    @staticmethod
    def _transform_y(flat_y: str) -> np.array:
        one_hot = OneHotEncoder()
        return one_hot.fit_transform(list(flat_y))

    def _build_model(self, embedding_dim=64, num_neurons=64) -> None:
        seq_len = self._window_size * 2 + 1
        vocab_size = len(self._x_index.vocabulary) + 1
        num_pos_tags = len(self._y_index)

        input_layer = Input(shape=(seq_len,))
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=seq_len)(input_layer)
        flatten_layer = Flatten()(embedding_layer)
        dense_layer = Dense(units=num_neurons,
                            activation='relu')(flatten_layer)
        predict_layer = Dense(units=num_pos_tags,
                              activation='softmax')(dense_layer)
        self._model = Model(input=input_layer, outputs=predict_layer)

    @staticmethod
    def _extract_nouns(char_list: list, predictions: np.ndarray,
                       text_lengths: list) -> list:
        raise NotImplementedError

    @property
    def train_history(self):
        return self._train_history

    def plot_train_history(self, save_name: str):
        for metric in ('loss', 'categorical_accuracy'):
            val_metric = 'val_' + metric
            plt.plot(self._train_history[metric], label=metric)
            plt.plot(self._train_history[val_metric], label=val_metric)

            if 'accuracy' in metric:
                metric = 'accuracy'

            plt.title(f'{metric} over epochs')
            plt.ylabel(metric)
            plt.xlabel('Epochs')
            plt.legend(loc="upper right")
            plt.savefig(f"{save_name}_{metric}.png")
            plt.clf()


class IntIndex:
    """
    Creates & preserves mapping of characters to integers.
    Can be created from an iterable that holds the vocabulary items,
    or from a json file that already contains the mappings.
    """

    def __init__(self, vocabulary=None, json_obj=None):
        """
        Initialize the index from an iterable of vocabulary or a json file.

        Parameters
        ----------
        vocabulary: Iterable
                    An iterable of items you want to map to integers.
                    If you wanted to index characters to integers, this vocabulary
                    would be a list of characters -- a string.

        json_obj: dict
                  If you wish to load the index from a previously saved index,
                  pass a dictionary into this parameter.
        """
        if json_obj:
            self._to_int = json_obj
            self._from_int = {value: key for key, value in
                              self._to_int.items()}
            self._vocabulary = set(self._to_int.keys())

        else:
            if not vocabulary:
                raise KeyError("You must load a json file or a vocabulary.")

            if not isinstance(vocabulary, Iterable):
                raise TypeError("Expected vocabulary to be an iterable,"
                                "but got type {}".format(type(vocabulary)))

            self._vocabulary = set(vocabulary)
            self._to_int = {}
            self._from_int = {}
            self._build_index()

    def _build_index(self) -> None:
        for n, obj in enumerate(self._vocabulary):
            self._to_int[obj] = n
            self._from_int[n] = obj

    def add_vocab(self, value) -> None:
        assert len(self._to_int) == len(self._from_int)
        index = len(self._to_int)
        self._to_int[value] = index
        self._from_int[index] = value

    def to_int(self, obj: str):
        """
        Converts a character into an integer.
        Parameters
        ----------
        obj:
            Object to be mapped into an integer.

        Returns
        -------
        int
            Integer corresponding to the input object.
        """
        try:
            return self._to_int[obj]
        except KeyError:
            return -1

    def from_int(self, integer: int):
        """
        Converts an integer into a character.
        Parameters
        ----------
        integer: int
                 Integer to be mapped into a character.

        Returns
        -------
        obj
            Object corresponding to the input integer.
        """
        try:
            return self._from_int[integer]
        except KeyError:
            return "\0"

    def to_json(self, file_name: str) -> None:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self._to_int, f)

    @classmethod
    def from_json(cls, file_name: str):
        with open(file_name, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
        return cls(json_obj=json_obj)

    @property
    def vocabulary(self):
        return self._vocabulary
