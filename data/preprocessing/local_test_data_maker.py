import unittest
from .data_maker import CorpusCleaner

"""
Run this test ONLY after you have
cloned sejong_corpus_cleaner and have downloaded the data 
from 국립국어원 database!
Otherwise, it will err.
"""


class DataMakerTest(unittest.TestCase):

    def setUp(self) -> None:
        self._x = "text"
        self._y = "tag"
        corpus_cleaner = CorpusCleaner()
        corpus_cleaner.generate()
        self._data = corpus_cleaner.data

    def tearDown(self) -> None:
        self._data = None

    def test_num_columns(self):
        data = self._data
        self.assertEqual(len(data[self._x]), len(data[self._y]))

    def test_row_lens(self):
        data = self._data
        for x_row, y_row in data[[self._x, self._y]].values:
            self.assertEqual(len(x_row), len(y_row))


if __name__ == "__main__":
    import os
    if 'sejong_corpus_cleaner' in os.listdir(os.getcwd()):
        unittest.main()
