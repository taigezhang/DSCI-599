import pandas as pd

# Define input and output file names
input_file = "cicddos2019_combined.csv"
output_file = "cicddos2019_cleaned.csv"

# Select relevant columns for Flow-to-Traffic Graph (FTG) representation
columns_to_keep = [
    "Source IP", "Destination IP", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Min", "Bwd Packet Length Max", "Fwd IAT Mean", "Flow Packets/s", "Flow Bytes/s",
    "SYN Flag Count", "Min Packet Length", "Max Packet Length", "Avg Packet Size", "Bwd Packet Length Mean", "Attack_Type"
]

# Read in chunks to reduce memory usage
chunk_size = 50000  # Process 50,000 rows at a time
chunks = []

print("🔹 Extracting features in chunks...")

# Process CSV in chunks
for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
    # Trim column names to remove extra spaces
    chunk.columns = chunk.columns.str.strip()

    # Keep only necessary columns
    chunk = chunk[columns_to_keep]

    # Convert numeric columns properly
    numeric_cols = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Protocol", "Fwd Packet Length Min", 
        "Bwd Packet Length Max", "Fwd IAT Mean", "Flow Packets/s", "Flow Bytes/s", "SYN Flag Count", 
        "Min Packet Length", "Max Packet Length", "Avg Packet Size", "Bwd Packet Length Mean"
    ]
    
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)  # Convert safely

    # Normalize numeric columns to [0,1] range
    chunk[numeric_cols] = chunk[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Append cleaned chunk to output CSV file
    chunk.to_csv(output_file, mode='a', index=False, header=not bool(chunks))
    chunks.append(chunk)

print("✅ Feature extraction completed. Saved as 'cicddos2019_cleaned.csv'")
