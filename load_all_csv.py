import pandas as pd
import os

# Define the dataset directory
DATASET_DIR = "raw-data"  # Ensure your CSV files are inside this folder

# Get a list of all CSV files
csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv") and f != "TFTP.csv"]

# Create an empty list to store DataFrames
df_list = []

# Load each CSV file and append it to the list
for file in csv_files:
    file_path = os.path.join(DATASET_DIR, file)
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)  # Load CSV
    df["Attack_Type"] = file.split(".")[0]  # Add attack type from filename
    df_list.append(df)

# Combine all DataFrames
full_df = pd.concat(df_list, ignore_index=True)

# Save combined dataset
full_df.to_csv("cicddos2019_combined.csv", index=False)
print("All CSV files loaded and saved as 'cicddos2019_combined.csv'")


