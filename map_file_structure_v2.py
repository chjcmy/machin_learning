
with open(r"c:/Users/UserK/machin_learning/D1_02_3_3_iris_classification_GD2601.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if len(line) > 200: continue # Skip long lines (likely image/html)
        if " = " in line and "(" in line:
            if "SVC" in line: print(f"SVC: Line {i+1}: {line}")
            if "LogisticRegression" in line: print(f"LogReg: Line {i+1}: {line}")
            if "DecisionTreeClassifier" in line: print(f"DecTree: Line {i+1}: {line}")
            if "VotingClassifier" in line: print(f"Voting: Line {i+1}: {line}")
            if "RandomForestClassifier" in line: print(f"RandForest: Line {i+1}: {line}")
            if "XGBClassifier" in line: print(f"XGB: Line {i+1}: {line}")
        if "accuracy_score" in line and "knn_acc" in line:
             print(f"KNN Acc: Line {i+1}: {line}")
