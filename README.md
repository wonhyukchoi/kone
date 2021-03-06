# KONE: Korean Open Noun Extractor

[![Travis CI](https://travis-ci.org/wonhyukchoi/kone.svg?branch=master)](https://travis-ci.org/wonhyukchoi/kone)

# 개요

인공신경망 레이어 2개 (Embedding 1개, dense 1개)만으로 원문에서 명사를 추출하는 모델입니다.

별도 사전이나 룰 없이, _순전히 머신러닝_으로만 학습된 모델입니다.

이 단순한 모델만으로 명사 [IOB 태깅](https://arxiv.org/abs/cs/9907006) 정확도가 무려 **98.9%** 나오며,  이를 바탕으로 어떠한 원문에서던지 명사를 추출할 수 있습니다.

| 예시 | 예시 |
|-|---|
| <span style="font-weight:normal">  뉴스  </span>|<span style="font-weight:normal">  '테니스 황제' 로저 페더러(39· 스위스· 4위)가 무릎 부상으로 수술대에 올랐다.  </span>|
| **명사** | **테니스, 황제, 로저, 페더러, 스위스, 무릎, 부상, 수술대** |
|                                                      |                                                              |
| <span style="font-weight:normal">디시인사이드</span> | <span style="font-weight:normal">기호를 괴상하게 많이 써서 다른 언어 프로그래머들이 피똥 싼다.</span> |
|**명사**  | **기호, 언어, 프로그래머, 피똥** |
|                                                      |                                                              |
| <span style="font-weight:normal">네이버 카페 </span>| <span style="font-weight:normal">저는 쇼팽을 좋아합니다. 실력은 안 되면서도 이미 수많은 에튀드들을 쳤습니다. </span>|
| **명사** | **쇼팽, 실력, 에튀드** |

Large모델조차도 모델 크기가 20Mb밖에 되지 않는, 작지만 강력한 모델입니다.

[사용법으로 바로가기](https://github.com/wonhyukchoi/kone#모듈-사용)

## 모델 아키텍처

모델은 단순히 Embedding layer 1개, Dense layer 1개만으로 구성되어 있습니다.

원문에서 글자를 window size 기준으로 선택하여, embedding 벡터화 한 뒤 flatten 하여 dense layer만 거치면 softmax output layer를 통하여서 해당 글자의 IOB 품사를 예측합니다.

아래 그림은 window size 2, embedding layer 3, neuron 10개 으로 "세"라는 글자의 품사를 예측하는 예시입니다.

<img src=".github/img/model.jpg" height="350">

## 학습 데이터

학습 데이터는 [국립국어원 세종 말뭉치](https://ithub.korean.go.kr/user/guide/corpus/guide1.do)로, 10만개 어절이 넘는 방대한 양의 데이터를 기반으로 학습하였습니다.

이 세종 코퍼스는 XML 형태로 되어있고, 여러 가지 오류도 많은데, [김현중님의 오픈 소스 코드](https://github.com/lovit/sejong_corpus_cleaner)를 참조하여 전처리 작업을 합니다.

# 사용 방법

## 모듈 사용

모듈로 코드에 삽입하여 쓰시고 싶다면 `usage_example.py`에 나와있는대로 사용하시면 됩니다.

```python
from kone.models import Kone
kone = Kone(KONE_WINDOW_SIZE)
kone.load_model(X_PATH, Y_PATH, MODEL_PATH)
nouns_list = kone.predict(TEXT_LIST)
```

이 때 물론 `KONE_WINDOW_SIZE, X_PATH, Y_PATH, MODEL_PATH, TEXT_LIST`는 기 선언한 변수여야 합니다.

## 학습 

본인이 스스로 학습해보고 싶다면, 아래 절차를 따라 하시면 됩니다.

## 학습 데이터 생성 방법

1. `cd data/preprocessing` 후 `git clone https://github.com/lovit/sejong_corpus_cleaner.git`을 해주세요.
2. 국립국어원 세종 말뭉치 다운로드 뒤, `data/preprocessing/sejong_corpus_cleaner/data/raw`에 넣어주세요.
3. `kone/make_data.sh` 실행
4. `data/train_data.csv` 가 등장합니다!

## 모델 학습 

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

## 모델 저장 

학습된 모델을 저장하시려면, 아래와 같이 저장하시면 됩니다.

```python
kone = Kone(...생략...)
kone.train(...생략...)
kone.save_model(X_INDEX_PATH, Y_INDEX_PATH, MODEL_WEIGHT_PATH)
```

원하시는 곳을 설정하여 저장하시면 됩니다. 

이때 `X_INDEX` 와 `Y_INDEX`는 `.json`으로 저장해야 되며, `MODEL_WEIGHT`는 `.h5`로 저장해야 됩니다.

이후에 모델이 저장되었으면 위 [모듈 사용](https://github.com/wonhyukchoi/kone#모듈-사용)에서처럼 모델을 로드하여 사용하시면 됩니다.



# 학습된 모델

이 repository에는 기 학습된 모델이 `models/`에 들어있습니다.

## Large 모델

Large 모델은 정확도를 위해 맞춘 모델로서, `weight` 는 21.4 Mb이고 정확도는 **98.9%** 입니다.


Large 모델 학습에 사용된 hyperparameter는 아래와 같습니다.

* numpy random seed = 47
* window size = 4
* epochs = 9
* batch size = 16384 
* embedding dim = 300
* num neurons = 100
* optimizer = rmsprop

이 hyperparameter로 생성된 모델의 정확도는 **98.9%** 입니다. (`sklearn`의 `train_test_split`로 생성된 dev test 기준)

Dense layer의 크기와 숫자를 늘리면 99.0% 까지 올릴 수 있으나, 얻는 정확도에 비해 모델 크기가 너무 크기 때문에 따로 탑재하지 않았습니다.



## Small 모델

Small 모델은 모델 크기에 중점을 두어서, 총 용량이 **2Mb**를 초과하지 않는 모델입니다.



Small 모델 학습에 사용된 hyperparameter는 아래와 같습니다.

* numpy random seed = 47
* window size = 3
* epochs = 15
* batch size  = 16384
* embedding dim = 32
* num neurons = 32
* optimizer = rmsprop

이 모델의 정확도는  **97.8%** 로 large 모델보다 정확도가 다소 떨어지지만, 대신 모델의 크기는 1/20 규모입니다. 

Google Colab에서 제공하는 Tesla K80 GPU 사용 시, epoch 당 학습 소요시간은 30초밖에 걸리지 않습니다.