
import json
import os

input_path = r'c:\Users\UserK\machin_learning\D1_03_3_4_boston_regression_GD2601.txt'
output_path = r'c:\Users\UserK\machin_learning\D1_03_3_4_boston_regression_GD2601.ipynb'

def add_comment(source_list, keyword, comment):
    """Adds a comment to the source list if the keyword is present."""
    for i, line in enumerate(source_list):
        if keyword in line and comment not in line:
            # Add comment before the line
            source_list.insert(i, f"# [ANNOTATION] {comment}\n")
            return

with open(input_path, 'r', encoding='utf-8') as f:
    notebook_data = json.load(f)

cells = notebook_data['cells']

for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        # 1. Train Test Split
        add_comment(source, 'train_test_split', 'Splitting data into 80% training and 20% testing sets to evaluate model generalization.')
        
        # 2. Linear Regression (Baseline)
        add_comment(source, 'LinearRegression()', 'Initializing the baseline Linear Regression model.')
        add_comment(source, 'lr.fit', 'Training the model on the training data.')
        
        # 3. Polynomial Features
        add_comment(source, 'PolynomialFeatures(degree=2)', 'Creating 2nd-degree polynomial features to capture non-linear relationships.')
        add_comment(source, 'PolynomialFeatures(degree=15)', 'Creating 15th-degree polynomial features. Note: High degrees often lead to overfitting.')
        
        # 4. Cross Validation
        add_comment(source, 'cross_val_score', 'Performing 5-fold cross-validation to check model stability across different data subsets.')
        
        # 5. MSE
        add_comment(source, 'mean_squared_error', 'Calculating Mean Squared Error (MSE) to quantify prediction accurancy (Lower is better).')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=2, ensure_ascii=False)

print(f"Successfully created annotated notebook at: {output_path}")
