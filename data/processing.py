import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("hope.csv")
print(df.shape)
print(df.columns)

df = df.loc[(df["Initial Temperature"] >= 298) & (df["Initial Temperature"] <= 500)]
df = df.loc[(df["Length-to-diameter Ratio"] >= 0.4) & (df["Length-to-diameter Ratio"] <= 3.0)]
df = df.loc[(df["Pressure"] >= 10.0e5) & (df["Pressure"] <= 40.0e5)]
print(df.shape)
df.to_csv("optimization_(cleaned).csv", index=False)