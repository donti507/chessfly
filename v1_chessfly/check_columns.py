from neuprint import Client, fetch_adjacencies

import os
TOKEN = os.environ["NEUPRINT_TOKEN"]
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=TOKEN)

conn_df, _ = fetch_adjacencies(sources=[12781], targets=None)
print("COLUMNS:", conn_df.columns.tolist())
print(conn_df.head())
