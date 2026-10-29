import pandas as pd

# Load the CSV file
df = pd.read_csv("pf1.csv")

df.drop(columns=["attack_type"], inplace=True)

df["label"] = df["label"].replace({"norm": 0, "anom": 1})

df.to_csv("pff.csv", index=False)

print(df.head())