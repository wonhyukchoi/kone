import os
import pandas as pd
from kone.models import Kone

model_dir = os.path.join(os.getcwd(), 'models')
x_index_path = os.path.join(model_dir, 'x_index.json')
y_index_path = os.path.join(model_dir, 'y_index.json')
weight_path = os.path.join(model_dir, 'model_weights.h5')

data_path = os.path.join(os.path.join(
    os.getcwd(), 'data'), 'sample.csv')
data = pd.read_csv(data_path)
x, y = data['text'], data['tag']

def train_and_predict():
    kone = Kone(window_size=3)
    kone.train(x=x, y=y, epochs=10)

    nouns = kone.predict(x)
    data['nouns'] = nouns
    data[['text', 'nouns']].to_csv('sample_result.csv',
                                   index=False, encoding='utf-8-sig')


def train_and_save():
    kone = Kone(window_size=3)
    kone.train(x=x, y=y, epochs=10)

    kone.save_model(x_index_name=x_index_path,
                    y_index_name=y_index_path,
                    weight_name=weight_path)


def load_and_predict():
    kone = Kone(window_size=3)
    kone.load_model(x_index_path=x_index_path,
                    y_index_path=y_index_path,
                    model_path=weight_path)

    nouns = kone.predict(x)
    data['nouns'] = nouns
    data[['text', 'nouns']].to_csv('sample_result.csv',
                                   index=False, encoding='utf-8-sig')


def train_save_load_predict():
    train_and_save()
    load_and_predict()


if __name__ == "__main__":
    train_save_load_predict()
