import pandas as pd

df = pd.read_csv("data/file.csv")

print(df.head(3))

# print(df.info())

# print(df.isnull().sum())

# print(df[df['price'] == 0])

# cols = ['price', 'model', 'mileage', 'engine_capacity', 'vehicle_age']
# for col in cols:
#     print(f"\n Rows where {col} == 0")
#     print (df[df[col] <=10])


print("\n Description of the columns:")
print(df.describe())