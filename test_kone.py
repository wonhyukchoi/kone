import os
import unittest
import pandas as pd
from random import randint
from kone.models import Kone, IntIndex


class KoneTest(unittest.TestCase):

    def setUp(self):
        kone = Kone(window_size=3)
        data_path = os.path.join(os.path.join(
            os.getcwd(), 'data'), 'train_data.csv')
        data = pd.read_csv(data_path).sample(1000)
        x, y = data['text'], data['tag']
        kone.train(x=x, y=y, epochs=10)
        predictions = kone.predict(x)
        self.x = x
        self.predictions = predictions

    def tearDown(self) -> None:
        self.x = None
        self.predictions = None

    def test_index_saving(self, str_len=10000, unicode_range=10000):
        """
        Test that the IntIndex class works properly.
        Creates a random string, places it inside IntIndex, saves it,
        loads it, and checks that nothing changed.

        Parameters
        ----------
        str_len: int
            parameter to create random string.
        unicode_range: int
            parameter to create random string.

        Returns
        -------
        None
            It's just a test; no returns!!
        """

        random_str = "".join(chr(randint(0, unicode_range))
                             for _ in range(str_len))
        random_char = random_str[randint(0, str_len)]

        index = IntIndex(vocabulary=random_str)
        indexed_int = index.to_int(random_char)

        temp_name = random_str[:10]
        index.to_json(temp_name)

        loaded_index = IntIndex.from_json(temp_name)
        os.remove(temp_name)
        loaded_char = loaded_index.from_int(indexed_int)

        msg = f"Saved index mismatch: \n" \
              f"had {random_char} originally for index {indexed_int}, \n" \
              f"but got {loaded_char} as result after loading"
        self.assertEqual(random_char, loaded_char, msg)

    def test_output_row_len(self):
        """
        Tests that if n inputs come in, then that we should have n outputs.
        """
        x_len = len(self.x)
        predict_len = len(self.predictions)
        msg = "Expected input and predictions length to be the same,\n" \
              "but got input length {} and predict length {}" \
              "".format(x_len, predict_len)
        self.assertEqual(x_len, predict_len, msg)

    def test_output_noun_len(self):
        """
        Tests that the length of nouns does not exceed the original text.
        """
        for x_row, predict_row in zip(self.x, self.predictions):
            text_len = len(x_row)
            noun_len = len(predict_row)
            msg = "Expected length of nouns to not exceed text length,\n" \
                  "but got input {} \nand nouns {}" \
                  "".format(x_row, predict_row)
            self.assertTrue(text_len >= noun_len, msg)

    def test_noun_inclusion(self):
        """
        Tests that all nouns in the predictions are included in the text.
        This includes multiple appearances of the same word.
        """
        for text_row, predict_row in zip(self.x, self.predictions):
            noun_list = predict_row.split(',')
            working_text = text_row
            for noun in noun_list:
                noun_idx = working_text.find(noun)
                msg = "Expected noun {} to be in the text," \
                      "but tests reveal that it did not exist.\n" \
                      "Text: {}".format(noun, text_row)
                self.assertTrue(noun_idx != -1, msg)
                working_text = working_text[noun_idx+1:]


if __name__ == "__main__":
    unittest.main()
