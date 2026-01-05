
with open(r"c:/Users/UserK/machin_learning/D1_02_3_3_iris_classification_GD2601.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines[1315:1325]):
        print(f"{1316+i}: {repr(line)}")
