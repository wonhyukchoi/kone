import unittest
from .data_maker import CorpusCleaner


class DataMakerTest(unittest.TestCase):

    def test_num_columns(self):
        corpus_cleaner = CorpusCleaner()
        corpus_cleaner.generate()
        data = corpus_cleaner.data
        self.assertTrue(data['x'] == data['y'])

    def test_row_lens(self):
        corpus_cleaner = CorpusCleaner()
        corpus_cleaner.generate()
        data = corpus_cleaner.data
        for x_row, y_row in zip(data[['x', 'y']].values):
            self.assertEqual(len(x_row), len(y_row))


if __name__ == "__main__":
    unittest.main()
