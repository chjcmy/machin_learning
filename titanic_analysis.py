
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Data Loading
train = pd.read_csv('타이타닉/train.csv')
test = pd.read_csv('타이타닉/test.csv')
submission = pd.read_csv('타이타닉/submission.csv')

# Configure matplotlib Korean font if necessary (Skipping for now as I can't check fonts)
plt.rcParams['axes.unicode_minus'] = False

# Combine for processing
data = pd.concat([train, test], sort=False, ignore_index=True)
data['TrainSplit'] = 'Train'
data.loc[len(train):, 'TrainSplit'] = 'Test'

# Feature Engineering
# Sex
data.loc[data['Sex']=='female', 'Sex'] = 0
data.loc[data['Sex']=='male', 'Sex'] = 1
data['Sex'] = data['Sex'].astype(int)

# Title
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
# Label Encode Title
le = LabelEncoder()
data['Title'] = le.fit_transform(data['Title'])

# Age
data['Age'] = data['Age'].fillna(data.groupby('Title')['Age'].transform('median'))
data['AgeBin'] = pd.cut(data['Age'], 5, labels=False) # Simplified binning

# Fare
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['FareLog'] = np.log1p(data['Fare'])

# Embarked
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# FamilySize
data['SibSp'] = data['SibSp'].fillna(0)
data['Parch'] = data['Parch'].fillna(0)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Visualization
print("Generating FamilySize Visualization...")
plt.figure(figsize=(10, 6))
sns.pointplot(x='FamilySize', y='Survived', data=data[data['TrainSplit']=='Train'])
plt.title('Survival Rate by Family Size')
plt.savefig('family_size_survival.png')
print("Saved family_size_survival.png")

# Checking correlations of numeric features
# Select numeric columns for correlation (Train only)
numeric_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FareLog', 'AgeBin', 'Title']
corr_data = data[data['TrainSplit']=='Train'][numeric_cols]

plt.figure(figsize=(12, 10))
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Saved correlation_matrix.png")

print("Analysis Complete.")
