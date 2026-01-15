
import pandas as pd

# Load data
df = pd.read_csv('Student_Performance_Dataset.csv')

# Correlations
numeric_df = df.select_dtypes(include=['number'])
correlations = numeric_df.corr()['Final_Percentage'].sort_values(ascending=False)
print("--- Correlations with Final Percentage ---")
print(correlations)
print("\n")

# Group by Parental Education
print("--- Mean Final Percentage by Parental Education ---")
print(df.groupby('Parental_Education')['Final_Percentage'].mean().sort_values(ascending=False))
print("\n")

# Group by Internet Access
print("--- Mean Final Percentage by Internet Access ---")
print(df.groupby('Internet_Access')['Final_Percentage'].mean().sort_values(ascending=False))
print("\n")

# Group by Extracurricular Activities
print("--- Mean Final Percentage by Extracurricular Activities ---")
print(df.groupby('Extracurricular_Activities')['Final_Percentage'].mean().sort_values(ascending=False))
print("\n")

# Group by Pass/Fail
print("--- Pass/Fail Counts ---")
print(df['Pass_Fail'].value_counts(normalize=True))
