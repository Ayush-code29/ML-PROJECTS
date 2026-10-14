import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('heart.csv')
# print(df.head())
# print(df.info())
# print(df['Cholesterol'].value_counts())
ch_mean = df.loc[df['Cholesterol'] != 0,'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean)

df['RestingBP'] = df['RestingBP'].round(2)
# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde = True)

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)


# plt.tight_layout()

# sns.countplot(x = df['Sex'],hue = df['HeartDisease'])
# plt.show()
df_encode = pd.get_dummies(df,drop_first=True)
# print(df_encode)
df_encode = df_encode.astype(int)
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
# print(df_encode)
X = df_encode.drop('HeartDisease',axis=1)
y = df_encode['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "KNN": KNeighborsClassifier(n_neighbors=5),
#     "Naive Bayes": GaussianNB(),
#     "Decision Tree": DecisionTreeClassifier(
#         random_state=42
#     ),
#     "SVM": SVC()
# }

# # 8. Train and evaluate every model
# results = []

# for name, model in models.items():

#     # Train model
#     model.fit(X_train, y_train)

#     # Predict on unseen test data
#     y_pred = model.predict(X_test)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)

#     results.append({
#         "Model": name,
#         "Accuracy (%)": round(accuracy * 100, 2)
#     })

#     print(f"\n{'=' * 45}")
#     print(f"Model: {name}")
#     
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy of this model is:')
acc = accuracy_score(y_pred,y_test)
print(f"{acc * 100:.2f}%")