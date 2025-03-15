import pandas as pd

# Load only the first row to get column names
file_path = "cicddos2019_combined.csv"

try:
    df = pd.read_csv(file_path, nrows=5, encoding="latin1")  # Load first 5 rows safely
    print("\nColumn Names:\n")
    print(df.columns.tolist())  # Print list of column names
except Exception as e:
    print(f"Error loading CSV: {e}")
