import pandas as pd
import re
import math
import Levenshtein

# Load the dataset
df = pd.read_csv("cleaned_file.csv")

# Function to count special characters
def count_special_chars(query):
    special_pattern = r"[\'\";#]|--|/\*.*?\*/''"
    return len(re.findall(special_pattern, query))

# Function to calculate Shannon entropy
def shannon_entropy(query):
    if not query:
        return 0
    prob = {char: query.count(char) / len(query) for char in set(query)}
    return -sum(p * math.log2(p) for p in prob.values())

# Function to calculate Levenshtein Distance from common benign queries
common_queries = [
    "SELECT * FROM users", 
    "SELECT name, age FROM employees", 
    "INSERT INTO orders VALUES (1, 'Product', 100)", 
    "UPDATE accounts SET balance = 500 WHERE id = 2"
]

def min_levenshtein_distance(query):
    return min(Levenshtein.distance(query, cq) for cq in common_queries)

# Apply functions to the dataset
df["special_char_count"] = df["payload"].astype(str).apply(count_special_chars)
df["shannon_entropy"] = df["payload"].astype(str).apply(shannon_entropy)
df["levenshtein_distance"] = df["payload"].astype(str).apply(min_levenshtein_distance)

# Save the updated dataset with new features
df.to_csv("updated_file.csv", index=False)

# Display the first few rows
print(df.head())