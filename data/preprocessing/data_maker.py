import os
import math
import logging
import pandas as pd
from tqdm import tqdm
from sejong_corpus_cleaner.sejong_corpus_cleaner.loader import Sentences
from sejong_corpus_cleaner.sejong_corpus_cleaner.simple_tag import to_simple_morphtags


class CorpusCleaner:
    def __init__(self, x="text", y="tag", logfile='data.log'):
        self._iob_tag_list = []
        self._data = pd.DataFrame(columns=[x, y])
        self._sentences = Sentences()
        logging.basicConfig(filename=logfile, level=logging.DEBUG)

    def generate(self):
        x_list, y_list = [], []
        for n, sentence in enumerate(tqdm(self._sentences)):
            iob_tags_list = []

            for eojeol, morpheme_and_tag in sentence:
                iob_tags = ['O'] * len(eojeol)
                iter_eojeol = eojeol

                for (morpheme, morphtag) in to_simple_morphtags(morpheme_and_tag):
                    if morphtag == 'Noun':

                        if eojeol.count(morpheme) < 1:
                            info = f"\n***어절***:{eojeol}\n" \
                                   f"***Morpheme***:{morpheme}\n" \
                                   f"***문장***: {sentence}"
                            logging.info(info)
                            continue

                        if 1 > len(morpheme):
                            debug = f"\n***어절***:{eojeol}\n" \
                                   f"***Morpheme***:{morpheme}\n" \
                                   f"***문장***: {sentence}"
                            logging.debug(debug)
                            continue

                        idx = iter_eojeol.find(morpheme)
                        iob_tags[idx] = 'B'

                        remaining = len(morpheme) - 1
                        iob_tags[idx+1: idx+len(morpheme)] = ['I'] * remaining
                        iter_eojeol = iter_eojeol[idx+len(morpheme):]

                iob_tags_list.append("".join(iob_tags))

            x = " ".join(sentence.eojeols)
            y = "O".join(iob_tags_list)
            assert len(x) == len(y)
            x_list.append(x)
            y_list.append(y)

        # TODO: probably better way
        data = {}
        for column, data_column in zip(self._data.columns, (x_list, y_list)):
            data[column] = data_column
        self._data = pd.DataFrame(data)

    def save(self, out_name="train_data.csv",
             path=os.getcwd(),
             dump_protocol=75000):

        num_dumps = math.ceil(len(self._data)/dump_protocol)
        data = self._data.dropna()
        for i in range(num_dumps):
            prefix = 'iter_' + str(i) + '_'
            file_name = os.path.join(path, prefix + out_name)
            index = i * dump_protocol
            data[index: index+dump_protocol].to_csv(file_name,
                                                    encoding='utf-8-sig')

    @property
    def data(self):
        return self._data


if __name__ == "__main__":
    corpus_cleaner = CorpusCleaner()
    corpus_cleaner.generate()
    save_path = os.path.join(os.getcwd(), 'processed')
    corpus_cleaner.save("data.csv", path=save_path)
