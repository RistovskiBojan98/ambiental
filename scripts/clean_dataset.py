import pandas as pd
import numpy as np
import sys

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ambiental.config import RAW_CSV_PATH

# Load dataset
df = pd.read_csv(RAW_CSV_PATH, sep=';')

# Remove empty last columns
df = df.iloc[:, :-2]

# Replace comma decimal with dot and convert to float
for col in df.columns[2:]:
    df[col] = df[col].astype(str).str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace -200 with NaN (missing values)
df.replace(-200, np.nan, inplace=True)

# Combine Date and Time into datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                               format='%d/%m/%Y %H.%M.%S')

# Drop original Date and Time
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Set datetime as index
df.set_index('Datetime', inplace=True)

# Sort just in case
df.sort_index(inplace=True)

# Quick check
print(df.head())
print(df.isna().sum())
