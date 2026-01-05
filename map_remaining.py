
with open(r"c:/Users/UserK/machin_learning/D1_02_3_3_iris_classification_GD2601.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 5300: continue
        line = line.strip()
        if len(line) > 200: continue
        if " = " in line and "(" in line:
            if "RandomForestClassifier" in line: print(f"RF: Line {i+1}: {line}")
            if "XGBClassifier" in line: print(f"XGB: Line {i+1}: {line}")
            if "BaggingClassifier" in line: print(f"Bagging: Line {i+1}: {line}")
            if "AdaBoostClassifier" in line: print(f"AdaBoost: Line {i+1}: {line}")
            if "GradientBoostingClassifier" in line: print(f"GBM: Line {i+1}: {line}")
        if "KFold" in line: print(f"KFold: Line {i+1}: {line}")
        if "cross_val_score" in line: print(f"CV Score: Line {i+1}: {line}")
        if "grid_search" in line or "GridSearchCV" in line: print(f"GridSearch: Line {i+1}: {line}")
