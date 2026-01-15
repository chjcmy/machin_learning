
import json
import os

# Paths
linear_path = r'c:\Users\UserK\machin_learning\D1_01_linear_regression_basics_GD2601.ipynb'
boston_path = r'c:\Users\UserK\machin_learning\D1_03_3_4_boston_regression_GD2601.ipynb'

def create_markdown_cell(title, why, important, focus):
    """Creates a markdown cell dictionary."""
    source_lines = [
        f"### {title}\n",
        "\n",
        f"**1. 이유 (Why):** {why}\n",
        "\n",
        f"**2. 중요 포인트 (Key Point):** {important}\n",
        "\n",
        f"**3. 주목할 점 (Focus):** {focus}\n"
    ]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def is_annotation(cell):
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and '이유 (Why)' in cell['source'][0]:
            return True
        if len(cell['source']) > 0 and '###' in cell['source'][0]:
             return True
    return False

def annotate_linear(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    new_cells = []
    for cell in notebook['cells']:
        if is_annotation(cell):
            continue
            
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            if "import numpy" in src and "LinearRegression" in src:
                new_cells.append(create_markdown_cell(
                    "1. 라이브러리 (도구) 준비",
                    "계산기(numpy), 그래프(plt), 모델(sklearn) 등 전문가 도구를 가져옵니다.",
                    "LinearRegression이 우리가 쓸 '직선 긋기' 모델입니다.",
                    "에러 없이 실행되는지 확인하세요."
                ))
            elif "np.random.rand" in src and "noise" in src:
                new_cells.append(create_markdown_cell(
                    "2. 가짜 데이터 만들기",
                    "정답(y=3x+5)을 아는 데이터를 만들어서 모델을 시험해봅니다.",
                    "noise(잡음)를 섞어서 현실 데이터처럼 만듭니다.",
                    "X(문제)와 y(정답)가 만들어집니다."
                ))
            elif "train_test_split" in src:
                new_cells.append(create_markdown_cell(
                    "3. 데이터 나누기 (모의고사 vs 수능)",
                    "모델이 답을 외우지 못하게 공부용(Train)과 시험용(Test)을 나눕니다.",
                    "test_size=0.2는 20%를 시험용으로 뺀다는 뜻입니다.",
                    "데이터가 4조각으로 잘 나뉘었는지 보세요."
                ))
            elif "model = LinearRegression()" in src:
                new_cells.append(create_markdown_cell(
                    "4. 모델 학습 (공부시키기)",
                    "모델에게 공부용 데이터(X_train, y_train)를 주고 규칙을 찾으라고 합니다.",
                    "fit() 명령어가 바로 학습을 시키는 주문입니다.",
                    "기울기(coef)와 절편(intercept)을 잘 찾았는지 확인하세요."
                ))
            elif "mean_squared_error" in src or "r2_score" in src:
                new_cells.append(create_markdown_cell(
                    "5. 채점 하기",
                    "숨겨둔 시험 문제(X_test)를 풀게 하고 점수를 매깁니다.",
                    "R^2 점수가 1.0에 가까울수록 천재 모델입니다.",
                    "Test R^2가 0.9 이상이면 성공입니다."
                ))
            elif "plt.scatter" in src and "plt.plot" in src:
                new_cells.append(create_markdown_cell(
                    "6. 눈으로 확인하기 (그래프)",
                    "숫자 점수만 믿지 말고 실제 그래프로 확인합니다.",
                    "빨간 선(모델)이 점들(데이터)을 잘 관통하고 있는지 보세요.",
                    "비슷하게 잘 그어졌다면 성공입니다."
                ))

        new_cells.append(cell)
    
    notebook['cells'] = new_cells
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Annotated Linear Regression: {path}")

def annotate_boston(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    new_cells = []
    for cell in notebook['cells']:
        if is_annotation(cell):
            continue
            
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            if "read_csv" in src and "sep=" in src:
                 new_cells.append(create_markdown_cell(
                    "1. 데이터 로드 (어려운 파일 읽기)",
                    "깔끔하지 않은 텍스트 데이터를 읽어서 표로 만듭니다.",
                    "sep='\\s+'는 공백이 뒤죽박죽 섞여 있어도 잘 자르라는 뜻입니다.",
                    "데이터 크기(shape)가 (506, 13)인지 확인하세요."
                ))
            elif "model = LinearRegression()" in src:
                 new_cells.append(create_markdown_cell(
                    "2. 기본 모델 (기준점 잡기)",
                    "가장 단순한 선형 회귀로 먼저 점수를 내봅니다.",
                    "이 점수가 우리의 '최소 기준'이 됩니다.",
                    "MSE(에러)가 얼마나 나오는지 기억해두세요."
                ))
            elif "PolynomialFeatures" in src:
                 new_cells.append(create_markdown_cell(
                    "3. 다항 회귀 (곡선 그리기)",
                    "데이터를 제곱하거나 곱해서 특성을 뻥튀기합니다.",
                    "degree=2는 2차 곡선까지 그린다는 뜻입니다.",
                    "Train 점수는 좋은데 Test 점수가 나빠지면(과적합) 위험합니다."
                ))
            elif "Ridge" in src or "Lasso" in src or "ElasticNet" in src:
                 new_cells.append(create_markdown_cell(
                    "4. 규제 모델 (고삐 죄기)",
                    "모델이 너무 복잡해지지 않게 패널티를 줍니다.",
                    "과적합을 막아 Test 점수를 올리는 것이 목표입니다.",
                    "기본 모델보다 성능이 좋아졌는지 확인하세요."
                ))
            elif "DecisionTreeRegressor" in src or "RandomForestRegressor" in src:
                 new_cells.append(create_markdown_cell(
                    "5. 고급 모델 (트리 기반)",
                    "선 긋기 대신 '스무고개' 방식으로 문제를 풉니다.",
                    "RandomForest나 XGBoost는 실무에서 가장 많이 쓰는 강력한 모델입니다.",
                    "에러(MSE)가 확 줄어드는 것을 감상하세요."
                ))

        new_cells.append(cell)
    
    notebook['cells'] = new_cells
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Annotated Boston Regression: {path}")

if __name__ == "__main__":
    annotate_linear(linear_path)
    annotate_boston(boston_path)
