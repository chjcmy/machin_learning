
with open(r"c:/Users/UserK/machin_learning/D1_02_3_3_iris_classification_GD2601.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line_content = line.strip()
        if "accuracy_score" in line_content and "import" in line_content:
             print(f"KNN Accuracy: Line {i+1}")
        if "SVC(kernel='rbf')" in line_content or "SVC(" in line_content:
             print(f"SVC: Line {i+1}: {line_content}")
        if "LogisticRegression(" in line_content:
             print(f"LogisticRegression: Line {i+1}: {line_content}")
        if "DecisionTreeClassifier(" in line_content:
             print(f"DecisionTree: Line {i+1}: {line_content}")
        if "VotingClassifier(" in line_content:
             print(f"Voting: Line {i+1}: {line_content}")
        if "RandomForestClassifier(" in line_content:
             print(f"RandomForest: Line {i+1}: {line_content}")
        if "XGBClassifier(" in line_content:
             print(f"XGBoost: Line {i+1}: {line_content}")
