# 보스턴 주택 가격 예측 (회귀 심화) 상세 가이드

이 가이드는 `D1_03_3_4_boston_regression_GD2601.ipynb` 노트북의 심화 내용을 "왜 하는지", "무엇이 중요한지", "어디를 봐야 하는지"로 정리했습니다.

---

## 1. 데이터 로드 및 전처리 (Manual Data Loading)
```python
raw_df = pd.read_csv(data_url, sep="\s+", ...)
data = np.hstack(...)
```

### 1) 왜 하는가? (Why)
- **날것의 데이터 다루기**: 깔끔하게 정리된 엑셀 파일만 있는 게 아닙니다. 가끔은 인터넷에서 글자 덩어리를 가져와서 우리가 표로 만들어야 할 때가 있습니다.
- 이 코드는 원본 텍스트 데이터를 읽어서, 짝수 줄과 홀수 줄을 합쳐 하나의 데이터로 만드는 고급 전처리 과정입니다.

### 2) 무엇이 중요한가? (Key Point)
- `sep="\s+"`: 띄어쓰기(공백)가 불규칙하게 있어도 잘 잘라내라는 뜻입니다.
- `np.hstack`: 찢어진 데이터 조각을 옆으로 붙여서 하나로 합치는 함수입니다.

### 3) 무엇을 봐야 하는가? (Focus)
- `data.shape`: (506, 13)인지 확인하세요. 506개의 집 데이터와 13개의 특징(방 개수, 범죄율 등)이 잘 들어왔는지 봅니다.

---

## 2. 기본 선형 회귀 (Baseline Model)
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### 1) 왜 하는가? (Why)
- **기준점(Baseline) 잡기**: 가장 단순한 모델로 먼저 성적을 냅니다.
- 아무리 복잡한 모델이라도 이 "기본 모델"보다는 점수가 잘 나와야 의미가 있습니다.

### 2) 무엇이 중요한가? (Key Point)
- `MSE`(에러)가 얼마인지 기억해둡니다. (예: Test MSE가 약 24~25 정도 나옵니다.)

---

## 3. 다항 회귀 (Polynomial Regression)
```python
pf = PolynomialFeatures(degree=2)
X_train_poly = pf.fit_transform(X_train)
```

### 1) 왜 하는가? (Why)
- **곡선 그리기**: 선형 회귀는 "직선"만 그을 수 있습니다. 하지만 현실은 곡선일 수 있습니다.
- 데이터를 제곱(`^2`)하거나 서로 곱해서 특성을 뻥튀기하면, 모델이 곡선을 그릴 수 있게 됩니다.

### 2) 무엇이 중요한가? (Key Point)
- `degree=2`: 2차 함수(곡선)까지 허용한다는 뜻입니다.
- 특성 개수가 13개에서 104개로 늘어났음을 확인하세요 (`X_train_poly.shape`). 데이터가 풍부해집니다.

### 3) 무엇을 봐야 하는가? (Focus)
- **Train 점수는 오르고 Test 점수는 떨어지는지?**
- `degree`를 너무 높이면(예: 15) Train 점수는 환상적이지만 Test 점수는 엉망이 되는 **과적합(Overfitting)** 현상을 주의 깊게 보세요.

---

## 4. 규제 모델 (Ridge, Lasso, ElasticNet)
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```

### 1) 왜 하는가? (Why)
- **고삐 죄기**: 다항 회귀로 특성을 너무 많이 만들면 모델이 너무 예민해져서 과적합(Overfitting)되기 쉽습니다.
- "규제(Regularization)"는 공부를 너무 달달 외우지 못하게 패널티를 줘서, 시험(Test)에서도 잘 보게 만드는 기법입니다.

### 2) 무엇이 중요한가? (Key Point)
- `alpha`: 규제의 강도입니다. 크면 클수록 모델을 단순하게 만듭니다. (패널티를 세게 줌)

### 3) 무엇을 봐야 하는가? (Focus)
- 그냥 선형회귀보다 Ridge나 Lasso를 썼을 때 Test MSE가 더 낮아지는지(성능이 좋아지는지) 확인하세요.

---

## 5. 고급 모델 (Tree Based Models)
```python
DecisionTreeRegressor, RandomForestRegressor, XGBRegressor
```

### 1) 왜 하는가? (Why)
- **더 똑똑한 학생들**: 선 긋기(회귀) 말고, "스무고개" 방식(트리)으로 정답을 맞추는 최신 모델들입니다.
- 보통 정형 데이터(엑셀 데이터)에서는 랜덤 포레스트(RandomForest)나 XGBoost가 성능이 제일 좋습니다.

### 2) 무엇이 중요한가? (Key Point)
- `RandomForest`: 나무 100그루를 심어서 투표하는 방식. 안정적이고 강력합니다.
- `XGBoost`: 틀린 문제만 골라서 집중적으로 다시 공부하는 방식. 성능 끝판왕입니다.

### 3) 무엇을 봐야 하는가? (Focus)
- **Test MSE가 획기적으로 줄어드는지 보세요.** (예: 25 -> 10~15 수준으로 떨어질 수 있습니다.)
