import pandas as pd
import numpy as np

df = pd.read_csv("data/file.csv")

# basic overview
# print("shape of dataset", df.shape) #rows x columns 
# print("Columns names in dataset", df.columns)
# print("data types", df.dtypes)

# preview data
# print("\nfirst 5 rows", df.head())

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

print(df['title'].head(20))
print(df['title'].dtype)