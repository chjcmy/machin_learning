
import json
import os

input_path = r'c:\Users\UserK\machin_learning\D1_02_3_3_iris_classification_GD2601.ipynb'
output_path = r'c:\Users\UserK\machin_learning\D1_02_3_3_iris_classification_GD2601.ipynb'

def create_markdown_cell(title, why, important, focus):
    """Creates a markdown cell dictionary with formatted content."""
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

with open(input_path, 'r', encoding='utf-8') as f:
    notebook_data = json.load(f)

new_cells = []
cells = notebook_data['cells']

# Avoid double annotation if run multiple times
def is_annotation(cell):
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and '이유 (Why)' in cell['source'][0]:
            return True
        if len(cell['source']) > 0 and '###' in cell['source'][0]:
             return True
    return False

for cell in cells:
    # Skip existing annotations to prevent duplicates if script is run twice
    if is_annotation(cell):
        continue
        
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        
        # 1. Imports
        if "import pandas as pd" in source_text and "load_iris" not in source_text:
            md_cell = create_markdown_cell(
                "1. 라이브러리 준비",
                "요리 도구를 꺼내는 단계입니다. 엑셀 기능을 하는 pandas, 계산기 numpy, 그림 그리는 matplotlib를 가져옵니다.",
                "가장 많이 쓰는 기본 도구들입니다.",
                "에러 없이 실행되면 성공입니다."
            )
            new_cells.append(md_cell)

        # 2. Load Data
        elif "load_iris()" in source_text:
            md_cell = create_markdown_cell(
                "2. 데이터 가져오기",
                "분석할 '붓꽃 데이터'를 로드합니다.",
                "sklearn 라이브러리에 내장된 연습용 데이터를 사용합니다.",
                "iris 변수에 데이터가 담깁니다."
            )
            new_cells.append(md_cell)
            
        # 3. DataFrame
        elif "pd.DataFrame" in source_text:
            md_cell = create_markdown_cell(
                "3. 데이터프레임 변환",
                "컴퓨터가 좋아하는 형태(Array)를 사람이 보기 편한 표(DateFrame)로 바꿉니다.",
                "이 'df' 변수가 앞으로 분석의 핵심이 됩니다.",
                "표 모양이 예쁘게 잘 나왔는지 확인하세요."
            )
            new_cells.append(md_cell)

        # 4. EDA (Info)
        elif "df.info()" in source_text:
            md_cell = create_markdown_cell(
                "4. 데이터 건강검진 (Info)",
                "데이터에 빠진 값(Null)은 없는지, 숫자가 맞는지 확인합니다.",
                "150개가 꽉 차있는지(Non-Null) 중요합니다.",
                "모두 'float64'(실수)인지 확인하세요."
            )
            new_cells.append(md_cell)
            
        # 5. Split
        elif "train_test_split" in source_text:
            md_cell = create_markdown_cell(
                "5. 학습용/시험용 나누기",
                "로봇이 문제와 답을 몽땅 외우는(과적합) 것을 막기 위해, 공부할 문제(Train)와 시험 문제(Test)를 나눕니다.",
                "test_size=0.2는 20%를 시험용으로 뺀다는 뜻입니다.",
                "X_train(공부용), X_test(시험용)의 개수가 잘 나뉘었는지 보세요."
            )
            new_cells.append(md_cell)
            
        # 6. Model Train
        elif "LogisticRegression()" in source_text:
            md_cell = create_markdown_cell(
                "6. 모델 학습 (공부시키기)",
                "로봇(모델)을 불러와서 문제(X_train)와 답(y_train)을 주고 공부시킵니다(fit).",
                "fit() 함수가 바로 '학습'을 시키는 명령어입니다.",
                "에러 없이 넘어가면 로봇이 규칙을 깨우친 것입니다."
            )
            new_cells.append(md_cell)
            
        # 7. Predict & Score
        elif "predict(" in source_text or "accuracy_score" in source_text:
             # Only add if not added recently to avoid spamming if they are in separate cells close by
             # Simple heuristic: Just add it.
             pass 

        # Special case for explicit predict/score if separate
        if "predict(X_test)" in source_text:
             md_cell = create_markdown_cell(
                "7. 시험 치기 (예측)",
                "숨겨뒀던 시험 문제(X_test)를 주고 정답을 맞춰보라고 합니다.",
                "predict() 함수의 결과가 로봇이 써낸 답안지입니다.",
                "결과 배열에 0, 1, 2(꽃의 종류)가 나옵니다."
            )
             new_cells.append(md_cell)

        if "accuracy_score" in source_text:
             md_cell = create_markdown_cell(
                "8. 채점 하기",
                "로봇이 쓴 답과 실제 정답(y_test)을 비교해 점수를 매깁니다.",
                "1.0이면 100점입니다.",
                "점수가 너무 낮으면 공부 방법이나 재료를 바꿔야 합니다."
            )
             new_cells.append(md_cell)


    new_cells.append(cell)

# Update notebook content
notebook_data['cells'] = new_cells

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1, ensure_ascii=False)

print(f"Successfully annotated notebook: {output_path}")
