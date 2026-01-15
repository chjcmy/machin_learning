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
        "# 상관관계 시각화 (막대 그래프 및 선 그래프)\n",
        "plt.figure(figsize=(14, 6))\n",
        "\n",
        "# 막대 그래프\n",
        "plt.subplot(1, 2, 1)\n",
        "sns.barplot(x=correlations.head(10).index, y=correlations.head(10).values, palette='viridis')\n",
        "plt.title('Top 10 Correlations with Final Percentage (Bar)')\n",
        "plt.xticks(rotation=45)\n",
        "\n",
        "# 선 그래프\n",
        "plt.subplot(1, 2, 2)\n",
        "sns.lineplot(x=correlations.head(10).index, y=correlations.head(10).values, marker='o', color='b')\n",
        "plt.title('Top 10 Correlations with Final Percentage (Line)')\n",
        "plt.xticks(rotation=45)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
    }

    nb['cells'].append(new_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully appended cell to {notebook_path}")
else:
    print(f"Notebook file not found at {notebook_path}")
