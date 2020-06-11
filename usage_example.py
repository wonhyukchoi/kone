import os
from kone.models import Kone


KONE_WINDOW_SIZE = 4
MODEL_DIR = os.path.join(os.getcwd(), 'models')
X_PATH = os.path.join(MODEL_DIR, 'large_x_index.json')
Y_PATH = os.path.join(MODEL_DIR, 'large_y_index.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'large_weights.h5')

SAMPLE_TEXTS = [
    "'테니스 황제' 로저 페더러(39· 스위스· 4위)가 무릎 부상으로 수술대에 올랐다.",
    "기호를 괴상하게 많이 써서 다른 언어 프로그래머들이 피똥 싼다.",
    "저는 쇼팽을 좋아합니다. 실력은 안 되면서도 이미 수많은 에튀드들을 쳤습니다."
]


def print_result(text_list, noun_list):
    for text, nouns in zip(text_list, noun_list):
        print(f"원문:{text}\n"
              f"명사:{nouns}",
              end='\n\n')


if __name__ == "__main__":
    kone = Kone(KONE_WINDOW_SIZE)
    kone.load_model(X_PATH, Y_PATH, MODEL_PATH)
    nouns_list = kone.predict(SAMPLE_TEXTS)

    print_result(SAMPLE_TEXTS, nouns_list)
