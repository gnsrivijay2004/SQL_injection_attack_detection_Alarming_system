import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("final dataset.csv")

# Count the number of benign (0) and malignant (1) labels
benign_count = (df["label"] == '0').sum()
malignant_count = (df["label"] == '1').sum()


# Display counts
print(f"Benign (0) count: {benign_count}")
print(f"Malignant (1) count: {malignant_count}")

df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5"], inplace=True)

# Remove non-numeric values from 'label' column
df_cleaned = df[['payload', 'label']].dropna()
df_cleaned = df_cleaned[df_cleaned['label'].str.strip().str.isnumeric()]

# Convert label to integer type
df_cleaned['label'] = df_cleaned['label'].astype(int)

# Remove any empty or whitespace-only payloads
df_cleaned = df_cleaned[df_cleaned['payload'].str.strip() != ""]



# Query length distribution
df_cleaned["query_length"] = df_cleaned["payload"].apply(len)

# Plot query length distribution
plt.figure(figsize=(10,5))
sns.histplot(df_cleaned["query_length"], bins=50, kde=True)
plt.title("Distribution of Query Lengths")
plt.xlabel("Query Length")
plt.ylabel("Frequency")
plt.xlim(0, 500)
plt.show()

# Check class balance
plt.figure(figsize=(6,4))
sns.countplot(x=df_cleaned["label"])
plt.title("Class Distribution")
plt.xlabel("Label (0 = Benign, 1 = Malignant)")
plt.ylabel("Count")
plt.show()

# Display basic statistics on query lengths
df_cleaned["query_length"].describe()


# Display cleaned dataset info and first few rows
df_cleaned_info = df_cleaned.info()
df_cleaned_head = df_cleaned.head()
df_cleaned_info, df_cleaned_head


# Display first few rows of the preprocessed dataset
print(df_cleaned.head())

df_cleaned.to_csv("cleaned_file.csv", index=False)