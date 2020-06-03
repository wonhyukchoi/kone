import collections
import json


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

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, *args, **kwargs):
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError


class IntIndex:
    """
    Preserves mapping of characters to integers.
    """
    def __init__(self, vocabulary=(), json_obj=None):
        if json_obj:
            self._to_int = json_obj
            self._from_int = {value: key for key, value in
                              self._to_int.items()}

        else:
            if not isinstance(vocabulary, collections.abc.Iterable):
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
    def read_json(cls, file_name: str):
        with open(file_name, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
        return cls(json_obj=json_obj)
