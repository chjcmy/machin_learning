import json
import os

notebook_path = r'c:\Users\UserK\machin_learning\Student_Performance_Analysis.ipynb'

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cell = {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 나이별 수학 및 과학 최적 학습 시간 분석\n",
        "\n",
        "# 학습 시간을 반올림하여 그룹화 (공부 시간 구간 생성)\n",
        "df['Study_Hours_Rounded'] = df['Study_Hours_Per_Day'].round(0)\n",
        "\n",
        "# 나이와 학습 시간 구간별 평균 점수 계산\n",
        "age_study_perf = df.groupby(['Age', 'Study_Hours_Rounded'])[['Math_Score', 'Science_Score']].mean().reset_index()\n",
        "\n",
        "# 각 나이별로 가장 높은 점수를 낸 학습 시간 찾기\n",
        "optimal_math = age_study_perf.loc[age_study_perf.groupby('Age')['Math_Score'].idxmax()]\n",
        "optimal_science = age_study_perf.loc[age_study_perf.groupby('Age')['Science_Score'].idxmax()]\n",
        "\n",
        "# 시각화\n",
        "plt.figure(figsize=(12, 6))\n",
        "\n",
        "plt.plot(optimal_math['Age'], optimal_math['Study_Hours_Rounded'], marker='o', label='Optimal Math Study Hours', linestyle='-', color='blue')\n",
        "plt.plot(optimal_science['Age'], optimal_science['Study_Hours_Rounded'], marker='s', label='Optimal Science Study Hours', linestyle='--', color='green')\n",
        "\n",
        "plt.title('Optimal Study Hours per Day by Age (for Max Score)')\n",
        "plt.xlabel('Age')\n",
        "plt.ylabel('Study Hours per Day')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.xticks(optimal_math['Age'].unique())\n",
        "plt.yticks(sorted(df['Study_Hours_Rounded'].unique()))\n",
        "plt.show()\n",
        "\n",
        "print(\"--- 나이별 최적 학습 시간 (수학) ---\")\n",
        "print(optimal_math[['Age', 'Study_Hours_Rounded', 'Math_Score']].to_string(index=False))\n",
        "print(\"\\n--- 나이별 최적 학습 시간 (과학) ---\")\n",
        "print(optimal_science[['Age', 'Study_Hours_Rounded', 'Science_Score']].to_string(index=False))"
       ]
    }

    nb['cells'].append(new_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully appended Optimal Study Time analysis cell to {notebook_path}")
else:
    print(f"Notebook file not found at {notebook_path}")
