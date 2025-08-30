import pandas as pd
from lib.utils import EMAIL_JSON_PATH

print("Loading email data-set...")
print(pd.__version__)
df_list = []
chunks = pd.read_json(EMAIL_JSON_PATH, chunksize=5000, lines=True)
for chunk in chunks:
    df_list.append(chunk)

df = pd.concat(df_list)
df["date"] = pd.to_datetime(df["date"], utc=True, errors='coerce')
print(f"Successfully loaded {len(df)} records.")