import os
import unittest
from random import randint
from kone.models import Kone, IntIndex


class KoneTest(unittest.TestCase):

    def test_index_saving(self, str_len=10000, unicode_range=10000):

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

    def test_kone(self):
        kone = Kone(window_size=3)

        self.assertEqual(kone, kone)

        kone = Kone(window_size=3)
        data_path = os.path.join(os.path.join(
            os.getcwd(), 'data'), 'sample.csv')
        data = pd.read_csv(data_path)
        x, y = data['text'], data['tag']
        kone.train(x=x, y=y, epochs=10)
        foo = (kone.predict(x))


if __name__ == "__main__":
    unittest.main()
