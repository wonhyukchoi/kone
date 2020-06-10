# KOrean Noun Extractor

[![Travis CI](https://travis-ci.org/wonhyukchoi/kone.svg?branch=master)](https://travis-ci.org/wonhyukchoi/kone)

## 개요

인공신경망 레이어 2개 (Embedding 1개, dense 1개)만으로 원문에서 명사를 추출하는 모델입니다.

이 단순한 모델만으로 명사 [IOB 태깅](https://arxiv.org/abs/cs/9907006) 정확도가 무려 **99.1%** 나오며,  이를 바탕으로 어떠한 원문에서던지 명사를 추출할 수 있습니다.

상당히 가벼운 모델이기 때문에 학습하는데 8 Gb 램, Nvidia GeForce 940M GPU 하나만 사용하더라도 학습 시간이 1시간만 소요됩니다.

[설명은 됐고, 어떻게 쓰는건데?](https://github.com/wonhyukchoi/kone#모듈%20사용)

### 모델 아키텍처

모델은 단순히 Embedding layer 1개, Dense layer 1개만으로 구성되어 있습니다.

### 학습 데이터

학습 데이터는 [국립국어원 세종 말뭉치](https://ithub.korean.go.kr/user/guide/corpus/guide1.do)로, 10만개 어절이 넘는 방대한 양의 데이터를 기반으로 학습하였습니다.

이 세종 코퍼스는 XML 형태로 되어있고, 여러 가지 오류도 많은데, [김현중님의 오픈 소스 코드](https://github.com/lovit/sejong_corpus_cleaner) 를 참조하여 전처리 작업을 합니다.



## 사용 방법

### 모듈 사용

모듈로 코드에 삽입하여 쓰시고 싶다면 `usage_example.py`에 나와있는대로 사용하시면 됩니다.

```python
from kone.models import Kone
kone = Kone(KONE_WINDOW_SIZE)
kone.load_model(X_PATH, Y_PATH, MODEL_PATH)
nouns_list = kone.predict(TEXT_LIST)
```

이 때 물론 `KONE_WINDOW_SIZE, X_PATH, Y_PATH, MODEL_PATH, TEXT_LIST`는 기 선언한 변수여야 합니다.

### 학습 

본인이 스스로 학습해보고 싶다면, 아래 절차를 따라 하시면 됩니다.

#### 학습 데이터 생성 방법

1. `cd data/preprocessing` 후 `git clone https://github.com/lovit/sejong_corpus_cleaner.git`을 해주세요.
2. 국립국어원 세종 말뭉치 다운로드 뒤, `data/preprocessing/sejong_corpus_cleaner/data/raw`에 넣어주세요.
3. `kone/make_data.sh` 실행
4. `data/train_data.csv` 가 등장합니다!

#### 모델 학습 

학습하려면 hyperparameter만 설정하여서 학습시켜주면 됩니다.

```python
kone = Kone(window_size=WINDOW_SIZE)
kone.train(x_data, y_data,
           epochs=EPOCHS,
           batch_size=BATCH_SIZE,
           embedding_dim=EMBEDDING_DIM,
           num_neurons=NUM_NEURONS,
           optimizer=OPTIMIZER)
```

Hyperparameter는 `WINDOW_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_NEURONS, OPTIMIZER`만 설정해주시면 됩니다.

#### 모델 저장 

학습된 모델을 저장하시려면, 아래와 같이 저장하시면 됩니다.

```python
kone = Kone(...생략...)
kone.train(...생략...)
kone.save_model(X_INDEX_PATH, Y_INDEX_PATH, MODEL_WEIGHT_PATH)
```

원하시는 곳을 설정하여 저장하시면 됩니다. 

이때 `X_INDEX` 와 `Y_INDEX`는 `.json`으로 저장해야 되며, `MODEL_WEIGHT`는 `.h5`로 저장해야 됩니다.

이후에 모델이 저장되었으면 위 [모듈 사용](https://github.com/wonhyukchoi/kone#모듈%20사용)에서처럼 모델을 로드하여 사용하시면 됩니다.



## 학습된 모델

이 repository에는 기 학습된 모델이 `models/`에 들어있습니다.

학습에 사용된 hyperparameter는 아래와 같습니다.

* numpy random seed = 47
* window size = 3
* epochs = 10
* batch size = 16384 
* embedding dim = 300
* num neurons = 100
* optimizer = rmsprop

이 hyperparameter로 생성된 모델의 validation accuracy는 **99.1%** 입니다. 