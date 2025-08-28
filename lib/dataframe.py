import pandas as pd
import pickle
from lib.utils import METADATA_PATH

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata_store = pickle.load(f)

# Create a Pandas DataFrame for efficient filtering
df = pd.DataFrame(metadata_store)
# Convert date column to datetime objects for proper filtering
df['date'] = pd.to_datetime(df['date'])
print(f"Successfully loaded {len(df)} records.")