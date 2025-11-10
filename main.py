import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/refined_file.csv")
# df = df[df["price"] != 0]
# df = df[df["vehicle_age"] != 0]

# basic overview
# print("shape of dataset", df.shape) #rows x columns 
print("Columns names in dataset", df.columns)
# print("data types", df.dtypes)

# preview data
print("\nfirst 5 rows", df.head())

# print("\n last 5 rows", df.tail())

# check missing values

# print("Missing values per column", df.isnull().sum())

# duplicates values

# print("number of duplicate rows", df.duplicated().sum())

# summary stats for numeric columns
# print("summary stats:\n", df.describe())

# summary stats for object column
# print("summary stats:\n", df.describe(include='object'))

# unique values per column

# for col in df.columns:
#     print(f"{col} - Unique values {df[col].nunique()}")

# check correaltion for numeric columns

# print("Correlation Matrix \n:", df.select_dtypes(include=[np.number]).corr())

# print(df['title'].head(20))
# print(df['title'].dtype)


# =======================================================================================

zero_counts = (df == 0).sum()

column_with_zeros = zero_counts[zero_counts > 0]

print("\n Columns with zero values")
print(column_with_zeros)

# ===========================================================================
# Save cleaned dataset to a new CSV file
# df.to_csv("data/refined_file.csv", index=False)

# print("✅ Cleaned dataset saved successfully as 'data/cleaned_file.csv'")



# ====================================Remove those values which have zero values=============================



