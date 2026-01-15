# 붓꽃(Iris) 데이터 분석 상세 가이드

이 가이드는 `D1_02_3_3_iris_classification_GD2601.ipynb` 노트북의 각 단계를 "왜 하는지", "무엇이 중요한지", "어디를 봐야 하는지"로 나누어 설명합니다.

---

## 1. 라이브러리 임포트 (Import Libraries)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
```

### 1) 왜 하는가? (Why)
- 요리에 비유하면 조리 도구(칼, 도마, 냄비)를 꺼내는 단계입니다.
- 파이썬 혼자서는 복잡한 표 계산이나 그래프 그리기를 못하기 때문에, 전문가들이 만들어둔 도구(라이브러리)를 빌려오는 것입니다.

### 2) 무엇이 중요한가? (Key Point)
- `pandas`: 엑셀처럼 표(Table)를 다루는 도구. 가장 많이 씁니다.
- `sklearn` (Scikit-learn): 머신러닝의 모든 것(데이터, 모델, 평가)이 들어있는 보물창고입니다.

### 3) 무엇을 봐야 하는가? (Focus)
- 에러 없이 실행되는지만 확인하면 됩니다.

---

## 2. 데이터 로드 및 확인 (Load Data)
```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
```

### 1) 왜 하는가? (Why)
- 분석할 재료(데이터)를 가져와서, 우리가 보기 편한 형태(`DataFrame`, 엑셀 표 모양)로 만드는 과정입니다.
- `load_iris()`가 주는 원본 데이터는 딕셔너리 형태라 눈에 잘 안 들어옵니다.

### 2) 무엇이 중요한가? (Key Point)
- `df`: 이제부터 모든 분석은 이 `df` 변수(DataFrame)를 가지고 합니다.
- `columns`: 각 열이 무엇을 의미하는지(꽃받침 길이, 꽃잎 길이 등) 알아아 합니다.
- `target`: 우리가 맞춰야 할 정답(붗꽃의 종류)입니다.

### 3) 무엇을 봐야 하는가? (Focus)
- `df.head()` 실행 결과에서 표가 예쁘게 잘 나왔는지 확인하세요.
- 데이터에 이상한 값(NaN 등)이 없는지 쓱 훑어보세요.

---

## 3. 데이터 탐색 (EDA)
```python
df.info()
df.describe()
```

### 1) 왜 하는가? (Why)
- 데이터의 건강 검진입니다. 데이터가 깨끗한지, 비어있는 값은 없는지, 숫자인지 문자인지 확인합니다.

### 2) 무엇이 중요한가? (Key Point)
- `info()`: "Null" 값이 있는지 보세요. 있으면 채워주거나 지워야 합니다. (이 데이터는 깨끗해서 없습니다.)
- `describe()`: 평균(mean), 최소(min), 최대(max)를 보고 데이터의 전반적인 크기를 파악합니다.

### 3) 무엇을 봐야 하는가? (Focus)
- `Non-Null Count`: 모두 150개로 꽉 차 있는지? (OK)
- `mean`: 평균값들이 너무 튀지 않는지? (OK)

---

## 4. 학습용/테스트용 데이터 분리 (Train/Test Split)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.2)
```

### 1) 왜 하는가? (Why)
- **가장 중요한 단계!** 로봇이 답지를 외워서 시험을 잘 보는 것(과적합)을 막기 위해, 공부할 문제(Train)와 시험 볼 문제(Test)를 강제로 찢어놓는 것입니다.

### 2) 무엇이 중요한가? (Key Point)
- `test_size=0.2`: 전체 150개 중 20%(30개)는 시험용으로 따로 빼두겠다는 뜻입니다.
- `random_state`: 데이터를 섞을 때 매번 똑같이 섞이게 해줍니다. (이게 없으면 실행할 때마다 결과가 달라져서 헷갈립니다.)

### 3) 무엇을 봐야 하는가? (Focus)
- `X_train`의 개수가 120개(80%), `X_test`가 30개(20%)로 잘 나뉘었는지 확인합니다.

---

## 5. 모델 학습 (Training)
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 1) 왜 하는가? (Why)
- 로봇(모델)에게 공부를 시키는 단계입니다. `fit`이 "학습해라!"라는 명령어입니다.
- 꽃잎의 길이/너비(`X`)와 꽃의 종류(`y`) 사이의 규칙을 찾아내서 머릿속에 저장합니다.

### 2) 무엇이 중요한가? (Key Point)
- `LogisticRegression`: 분류 문제(이거냐 저거냐 맞추기)에서 가장 기본이 되는 모델입니다. 이름은 '회귀'지만 사실은 '분류기'입니다.
- `fit(문제, 정답)`: 꼭 훈련용 데이터(`train`)를 줘야 합니다. 시험용(`test`)을 주면 **커닝**입니다.

### 3) 무엇을 봐야 하는가? (Focus)
- 실행했을 때 에러가 안 나면 성공입니다. 학습은 순식간에 끝납니다.

---

## 6. 예측 및 평가 (Prediction & Evaluation)
```python
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
```

### 1) 왜 하는가? (Why)
- 공부를 잘했는지 시험(`X_test`)을 치고, 채점(`accuracy_score`)을 하는 단계입니다.

### 2) 무엇이 중요한가? (Key Point)
- `predict`: 시험 문제를 보고 답을 써내는 과정입니다.
- `accuracy_score`: 로봇이 쓴 답(`y_pred`)과 진짜 정답(`y_test`)을 비교해서 점수를 매깁니다.

### 3) 무엇을 봐야 하는가? (Focus)
- 점수(Score)가 1.0에 가까울수록 좋습니다. (1.0 = 100점).
- 만약 점수가 너무 낮으면(0.5 이하), 데이터 전처리가 잘못되었거나 모델이 너무 단순한 것입니다.
