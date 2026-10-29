import pandas as pd

# Load the CSV file
df = pd.read_csv("payload_full.csv")

# Drop the unwanted column (replace 'column_to_drop' with the actual column name)
df.drop(columns=["length"], inplace=True)

# Keep only rows where 'attack_type' is 'norm' or 'sqli'
df = df[df["attack_type"].isin(["norm", "sqli"])]

# Save the modified DataFrame back to CSV
df.to_csv("pf1.csv", index=False)

# Display the updated DataFrame
print(df.head())
