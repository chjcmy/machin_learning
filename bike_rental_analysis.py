
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_bike_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n--- Data Structure ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    if missing.sum() == 0:
        print("No missing values found.")
        
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Correlation with Target (Rentals) ---")
    # Assuming 'Rentals' is the target based on the notebook content viewing
    # The notebook mentioned `Rentals` (일일 대여량)
    # Let's check if 'Rentals' column exists, otherwise we'll print columns and ask/infer.
    if 'Rentals' in df.columns:
        corr = df.corr(numeric_only=True)['Rentals'].sort_values(ascending=False)
        print(corr)
    elif 'cnt' in df.columns: # Common name in bike sharing dataset
        print("Column 'Rentals' not found, using 'cnt' as target if available.")
        corr = df.corr(numeric_only=True)['cnt'].sort_values(ascending=False)
        print(corr)
    else:
        print("Target column 'Rentals' not found. Available columns:")
        print(df.columns.tolist())

if __name__ == "__main__":
    analyze_bike_data('c:/Users/UserK/machin_learning/day_02.csv')
