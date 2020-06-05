import math
import pandas as pd
from sejong_corpus_cleaner.sejong_corpus_cleaner.loader import Sentences


class CorpusCleaner:
    def __init__(self, x="raw", y="nouns", dump_protocol=75000):
        self._data = pd.DataFrame(columns=[x, y])
        self._sentences = Sentences()

    def generate(self):
        raise NotImplementedError

    def save(self, out_name="train_data.csv", dump_protocol=75000):
        num_dumps = math.ceil(len(self._data)/dump_protocol)
        for i in range(num_dumps):
            prefix = 'iter_' + str(i) + '_'
            index = i * dump_protocol
            self._data[index: index+dump_protocol].to_csv(prefix+out_name)

    @property
    def data(self):
        return self._data


if __name__ == "__main__":
    pass
