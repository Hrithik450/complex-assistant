from lib.utils import VECTOR_DATA_PATH, EMAIL_JSON_PATH
import polars as pl
import faiss

# Load the faiss data
print("Loading faiss index...")
index = faiss.read_index(VECTOR_DATA_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
print("Loading vector data")

# Load the email jsonl data
print("Loading emails data-set")
df = pl.read_ndjson(EMAIL_JSON_PATH)
df = df.with_columns(
    pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).dt.replace_time_zone("UTC").alias("date")
)
print(f"Successfully loaded {df.height} records.")