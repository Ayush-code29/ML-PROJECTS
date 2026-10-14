import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('house-prices.csv')
# print(df.head())
# print(df.info())

# numerical_cols = [
#     'Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers'
# ]

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()

# for i, col in enumerate(numerical_cols):
#     sns.histplot(data=df, x=col, kde=True, ax=axes[i])
#     axes[i].set_title(f'Distribution of {col}')

# axes[-1].set_visible(False)

# plt.tight_layout()
# plt.show()
# cols = ['Bedrooms', 'Bathrooms', 'Offers']

# for col in cols:
#     print(f"\nValue counts for {col}:")
#     print(df[col].value_counts().sort_index())
df['Brick'] = df['Brick'].map({
    'No': 0,
    'Yes': 1
})
df = pd.get_dummies(
    df,
    columns=['Neighborhood'],
    dtype=int
)

# Features and target
X = df.drop(['Price', 'Home'], axis=1)
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]  # Number of test samples
p = X_test.shape[1]  # Number of input features

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("R2 Score:", r2)
print("Adjusted R2 Score:", adjusted_r2)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print(df.head())
