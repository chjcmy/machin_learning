
with open(r"c:/Users/UserK/machin_learning/D1_02_3_3_iris_classification_GD2601.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "KNeighborsClassifier" in line:
            print(f"Line {i+1}: {line.strip()}")
