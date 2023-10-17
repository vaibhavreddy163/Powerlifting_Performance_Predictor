import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/vaibhavreddy/Documents/Scripts/Powerlifting/data/openpowerlifting.csv')

# Data Cleaning and Preprocessing
# Drop rows where 'TotalKg' is NaN
df_cleaned = df.dropna(subset=['TotalKg'])

# Fill missing 'Age' values with the median age
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)

# Fill missing 'BodyweightKg' values with the median body weight
df_cleaned['BodyweightKg'].fillna(df_cleaned['BodyweightKg'].median(), inplace=True)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('/Users/vaibhavreddy/Documents/Scripts/Powerlifting/data/cleaned_openpowerlifting.csv', index=False)

# Exploratory Data Analysis (EDA)
# Summary statistics of key columns
print(df_cleaned[['Age', 'BodyweightKg', 'TotalKg']].describe())

# Visualization: Distribution of total weight lifted across different age groups
sns.histplot(df_cleaned, x='TotalKg', hue='Sex', element='step', stat='density', common_norm=False)
plt.title('Distribution of Total Weight Lifted by Sex')
plt.xlabel('Total Weight Lifted (Kg)')
plt.ylabel('Density')
plt.show()
