import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the cleaned dataset
df_cleaned = pd.read_csv('/Users/vaibhavreddy/Documents/Scripts/Powerlifting/data/cleaned_openpowerlifting.csv')

# Data Preparation
features = ['Age', 'BodyweightKg', 'Sex']
target = 'TotalKg'
X = df_cleaned[features].copy()
y = df_cleaned[target]

# Encoding the 'Sex' column
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
