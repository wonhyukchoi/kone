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
    def __init__(self, window_size: int, delim=','):
        super().__init__()
        self._window_size = window_size
        self._x_index = None
        self._y_index = None
        self._model = None
        self._train_history = None
        self._delim = delim

    def train(self, x: Sequence, y: Sequence,
              padding_symbol='<p>',
              epochs=1, batch_size=1 << 14,
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
        predicted_pos = [self._y_index.from_int(predicted_y) for predicted_y in y_hat]

        assert len(predicted_pos) == len(flat_x_chars), f"Input and output lengths must match." \
                                                        f"This should never happen, though."

        noun_list = self._extract_nouns_batch(char_list=flat_x_chars,
                                              predicted_pos=predicted_pos,
                                              text_lengths=text_lengths)

        return noun_list

    def save_model(self,
                   x_index_name="x_index.json",
                   y_index_name="y_index.json",
                   weight_name="weights.h5"):
        self._x_index.to_json(file_name=x_index_name)
        self._y_index.to_json(file_name=y_index_name)
        self._model.save(weight_name)

    def load_index(self, index_path, index_for='x'):
        index_for = index_for.lower().strip()
        if index_for == 'x':
            self._x_index = IntIndex.from_json(index_path)
        elif index_for == 'y':
            self._y_index = IntIndex.from_json(index_path)
        else:
            raise KeyError("Expected index for either x or y,"
                           "but got {}".format(index_for))

    def load_weights(self, model_path):
        self._model = load_model(model_path)

    def load_model(self, x_index_path, y_index_path, model_path):
        """ Method to retain consistency with API syntax. """
        self.load_index(index_path=x_index_path, index_for='x')
        self.load_index(index_path=y_index_path, index_for='y')
        self.load_weights(model_path=model_path)

    def _transform_x(self, text_list: Iterable,
                     padding_symbol='<p>') -> dict:
        flat_x, text_lengths, matrix_list = "", [], []
        padding = [self._x_index.to_int(padding_symbol)] * self._window_size
        seq_len = len(padding) * 2 + 1

        for text in text_list:
            flat_x += text
            text_lengths.append(len(text))

            numericized = [self._x_index.to_int(char) for char in text]
            padded_nums = padding + numericized + padding

            for i in range(len(text)):
                matrix_list.append(padded_nums[i: i + seq_len])

        matrix = np.array(matrix_list, dtype='int16')
        return dict(flat_x=flat_x, text_lengths=text_lengths,
                    matrix=matrix)

    def _transform_y(self, flat_y: str) -> np.array:
        one_hot = OneHotEncoder()
        y_array = np.array([self._y_index.to_int(char) for char in flat_y])
        transformed_y = one_hot.fit_transform(y_array.reshape(-1, 1)).toarray()
        return transformed_y.astype('int8')

    def _build_model(self, embedding_dim=64, num_neurons=64) -> None:
        seq_len = self._window_size * 2 + 1
        vocab_size = len(self._x_index.vocabulary) + 1
        num_pos_tags = len(self._y_index.vocabulary)

        input_layer = Input(shape=(seq_len,))
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=seq_len)(input_layer)
        flatten_layer = Flatten()(embedding_layer)
        dense_layer = Dense(units=num_neurons,
                            activation='relu')(flatten_layer)
        predict_layer = Dense(units=num_pos_tags,
                              activation='softmax')(dense_layer)
        self._model = Model(inputs=input_layer, outputs=predict_layer)

    def _extract_nouns_batch(self,
                             char_list: list, predicted_pos: list,
                             text_lengths: list) -> list:
        noun_list = []
        text_start = 0
        for length in text_lengths:
            text_end = text_start + length
            text_row = char_list[text_start: text_end]
            pos_row = predicted_pos[text_start: text_end]
            noun_list.append(self._extract_nouns(text_row, pos_row))

            text_start = text_end

        return noun_list

    def _extract_nouns(self, text: list, pos_list: list,
                       begin='B', inside='I',
                       other='O') -> str:
        """
        Refer to the wikipedia page for more information on IOB tagging.
        https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
        From the pos tags and original text, extracts nouns
        that are more than just a single character long.

        Parameters
        ----------
        text: list
            A list of characters of the text string from which to extract nouns
        pos_list: list
            A list of pos tags for each character
        begin: char
            Symbol for the IOB 'Beginning' tag
        inside: char
            Symbol for the IOB 'Inside' tag
        other: char
            Symbol for "Other" (non-noun) tags

        Returns
        -------
        nouns: list
            List of noun strings extracted from the text and pos tags

        """
        nouns = []
        noun_string = ''
        for char, pos_tag in zip(text, pos_list):

            if pos_tag == begin:
                if len(noun_string) == 0:
                    noun_string += char
                else:
                    noun_string = ""

            elif pos_tag == inside:
                if len(noun_string) >= 1:
                    noun_string += char

            elif pos_tag == other:
                if len(noun_string) >= 2:
                    nouns.append(noun_string)
                noun_string = ""

            else:
                raise KeyError("Found illegitimate tag {}".format(pos_tag))

        # Since last iob tag could have been the inside tag
        if len(noun_string) >= 2:
            nouns.append(noun_string)

        nouns_as_str = self._delim.join(nouns)
        return nouns_as_str

    @property
    def train_history(self):
        return self._train_history

    def plot_train_history(self, save_name: str):
        for metric in ('loss', 'categorical_accuracy'):
            val_metric = 'val_' + metric
            plt.plot(self._train_history.history[metric], label=metric)
            plt.plot(self._train_history.history[val_metric], label=val_metric)

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
