import pandas as pd
df = pd.read_csv("../../../docs/Health_Data/DIAGNOSIS.csv", nrows=0)
print(df.columns.tolist())
