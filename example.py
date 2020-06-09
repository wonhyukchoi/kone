import os
import pandas as pd
from kone.models import Kone


def train_and_predict():
    kone = Kone(window_size=3)
    data_path = os.path.join(os.path.join(
        os.getcwd(), 'data'), 'sample.csv')

    data = pd.read_csv(data_path)
    x, y = data['text'], data['tag']
    kone.train(x=x, y=y, epochs=10)

    nouns = kone.predict(x)
    data['nouns'] = nouns
    del data['tag']
    data.to_csv('sample_result.csv', index=False, encoding='utf-8-sig')


def load_and_predict():
    raise NotImplementedError


if __name__ == "__main__":
    train_and_predict()
