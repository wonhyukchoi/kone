# KOrean Noun Extractor

[![Travis CI](https://travis-ci.org/wonhyukchoi/kone.svg?branch=master)](https://travis-ci.org/wonhyukchoi/kone)

**_현재 코드는 사용 가능하나, 아직 모델 로딩과 추가적인 학습 등 미완성 repository 입니다._**

## 개요

인공신경망 레이어 2개 (Embedding 1개, dense 1개)만으로 원문에서 명사를 추출하는 모델입니다.

### 학습 데이터

학습 데이터는 [국립국어원 세종 말뭉치](https://ithub.korean.go.kr/user/guide/corpus/guide1.do)로, 10만개 어절이 넘는 방대한 양의 데이터를 기반으로 학습하였습니다.

이 세종 코퍼스는 XML 형태로 되어있고, 여러 가지 오류도 많은데, [김현중님의 오픈 소스 코드](https://github.com/lovit/sejong_corpus_cleaner) 를 참조하여 전처리 작업을 합니다.

### 모델 아키텍처

모델은 단순히 Embedding layer 1개, Dense layer 1개만으로 구성되어 있습니다.



## 사용 방법

### 예시

`example.py`에 학습 후 예측과, 모델 로드 후 예측 두 가지 코드를 준비 해놓았습니다.

### 모듈 사용

모듈로 코드에 삽입하여 쓰시고 싶다면 아래와 같이 사용하시면 됩니다.

```python
from kone.models import Kone
kone = Kone(window_size)
kone.load_model(x_index_path, y_index_path, model_path)
noun_list = kone.predict(texts)
```

이 때 물론 `window_size, x_index_path, y_index_path, model_path`는 기 선언한 변수여야 합니다.



## 학습

본인이 스스로 학습해보고 싶다면, 아래 절차를 따라 하시면 됩니다.

### 학습 데이터 생성 방법

1. `cd data/preprocessing` 후 `git clone https://github.com/lovit/sejong_corpus_cleaner.git`을 해주세요.
2. 국립국어원 세종 말뭉치 다운로드 뒤, `data/preprocessing/sejong_corpus_cleaner/data/raw`에 넣어주세요.
3. `kone/make_data.sh` 실행
4. `data/train_data.csv` 가 등장합니다!

### 학습

**_WIP_**