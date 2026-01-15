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
        "# 나이별 수학 및 과학 점수 분석\n",
        "age_performance = df.groupby('Age')[['Math_Score', 'Science_Score']].mean()\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.lineplot(data=age_performance, marker='o')\n",
        "plt.title('Average Math and Science Scores by Age')\n",
        "plt.ylabel('Average Score')\n",
        "plt.xlabel('Age')\n",
        "plt.grid(True)\n",
        "plt.legend(['Math Score', 'Science Score'])\n",
        "plt.xticks(age_performance.index)\n",
        "plt.show()\n",
        "\n",
        "# 점수가 만나는 지점 또는 격차가 가장 작은 지점 확인\n",
        "age_performance['Gap'] = abs(age_performance['Math_Score'] - age_performance['Science_Score'])\n",
        "min_top = age_performance.sort_values('Gap').head(1)\n",
        "print(\"수학/과학 점수 격차가 가장 작은 나이 (중간점):\")\n",
        "print(min_top[['Math_Score', 'Science_Score', 'Gap']])"
       ]
    }

    nb['cells'].append(new_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully appended Math/Science analysis cell to {notebook_path}")
else:
    print(f"Notebook file not found at {notebook_path}")
