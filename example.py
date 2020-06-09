import os
import pandas as pd
from kone.models import Kone


if __name__ == "__main__":
    kone = Kone(window_size=3)
    data_path = os.path.join(os.path.join(
        os.getcwd(), 'data'), 'sample.csv')
    data = pd.read_csv(data_path)
    x, y = data['text'], data['tag']
    kone.train(x=x, y=y)
