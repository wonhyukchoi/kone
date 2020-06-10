import os
from kone.models import Kone


KONE_WINDOW_SIZE = 3
MODEL_DIR = os.path.join(os.getcwd(), 'models')
X_PATH = os.path.join(MODEL_DIR, 'x_index.json')
Y_PATH = os.path.join(MODEL_DIR, 'y_index.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_weights.h5')

SAMPLE_TEXTS = [
    "행복한 즐거운 세상입니다!",
    "오토바이 타고 가자"
]


def print_result(text_list, noun_list):
    for text, nouns in zip(text_list, noun_list):
        print(f"Text:{text}\n"
              f"Nouns:{nouns}",
              end='\n\n')


if __name__ == "__main__":
    kone = Kone(KONE_WINDOW_SIZE)
    kone.load_model(X_PATH, Y_PATH, MODEL_PATH)
    nouns_list = kone.predict(SAMPLE_TEXTS)

    print_result(SAMPLE_TEXTS, nouns_list)
