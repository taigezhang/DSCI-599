import pandas as pd

# Load only first 5 rows
df = pd.read_csv("cicddos2019_cleaned.csv", nrows=5)

# Display structure
print("\n🔹 First 5 Rows of cicddos2019_cleaned.csv:\n")
print(df.head())

print("\n🔹 Column Names:\n", df.columns.tolist())  # Print column names
