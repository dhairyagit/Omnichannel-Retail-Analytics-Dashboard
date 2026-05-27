import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv('datasets/Omnichannel Retail Data.csv')

print("\n========== DATASET LOADED ==========")
print(df.head())

# =====================================================
# DATASET INFO
# =====================================================

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

# =====================================================
# CHECK MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\n========== DUPLICATES ==========")

duplicate_rows = df.duplicated().sum()

print("Duplicate Rows:", duplicate_rows)

df = df.drop_duplicates()

print("Duplicates Removed Successfully")

# =====================================================
# OUTLIER DETECTION
# =====================================================

print("\n========== OUTLIER DETECTION ==========")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"{col} --> {len(outliers)} outliers")

# =====================================================
# CHECK CATEGORICAL INCONSISTENCIES
# =====================================================

print("\n========== CATEGORICAL VALUES ==========")

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:

    print(f"\nColumn: {col}")
    print(df[col].unique())

# =====================================================
# VALIDATE COLUMN RELATIONSHIPS
# =====================================================

print("\n========== VALIDATION CHECKS ==========")

# Selling Price > Market Price

invalid_price = df[
    df['Selling Price'] > df['Market Price']
]

print(
    "Selling Price > Market Price:",
    len(invalid_price)
)

# Discounts > Market Price

invalid_discount = df[
    df['Discounts'] > df['Market Price']
]

print(
    "Discounts > Market Price:",
    len(invalid_discount)
)

# Reorder Point > Stock Levels

invalid_reorder = df[
    df['Reorder Point'] > df['Stock Levels']
]

print(
    "Reorder Point > Stock Levels:",
    len(invalid_reorder)
)

# Return Rate > 100

invalid_return = df[
    df['Return Rate (%)'] > 100
]

print(
    "Return Rate > 100:",
    len(invalid_return)
)

# =====================================================
# CLEAN SALES / INVENTORY ANOMALIES
# =====================================================

print("\n========== CLEANING ANOMALIES ==========")

# Fix Selling Price

df['Selling Price'] = np.where(
    df['Selling Price'] > df['Market Price'],
    df['Market Price'],
    df['Selling Price']
)

# Fix Discounts

df['Discounts'] = np.where(
    df['Discounts'] > df['Market Price'],
    df['Market Price'],
    df['Discounts']
)

# Fix Reorder Point

df['Reorder Point'] = np.where(
    df['Reorder Point'] > df['Stock Levels'],
    df['Stock Levels'],
    df['Reorder Point']
)

print("Anomalies Cleaned Successfully")

# =====================================================
# ENCODE CATEGORICAL DATA
# =====================================================

print("\n========== ENCODING ==========")

label_cols = [
    'Promotions',
    'Demand Trend',
    'Customer Segments'
]

le = LabelEncoder()

for col in label_cols:

    df[col] = le.fit_transform(df[col])

# One Hot Encoding

df = pd.get_dummies(
    df,
    columns=['Category']
)

print("Encoding Completed")

# =====================================================
# SAVE CLEANED DATASET
# =====================================================

output_file = 'cleaned_omnichannel_retail_data.csv'

df.to_csv(
    output_file,
    index=False
)

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n========== FINAL OUTPUT ==========")

print(
    "Cleaned dataset saved as:",
    output_file
)

print(
    "Final Shape:",
    df.shape
)

print("\n========== TASK COMPLETED ==========")