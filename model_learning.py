import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv('loan_approval_dataset.csv')
df_ml = df.copy()
df_ml.drop(columns=['loan_id'], inplace=True)
# Remove leading spaces from column names
df_ml.rename(columns=lambda x: x.strip(), inplace=True)
# print(df_ml.columns)
# Display the updated DataFrame
# print(df_ml.head())


label_encoder = LabelEncoder()

# Apply label encoding to the 'education' column
df_ml['education'] = label_encoder.fit_transform(df_ml['education'])

# Apply label encoding to the 'self_employed' column
df_ml['self_employed'] = label_encoder.fit_transform(df_ml['self_employed'])

# Apply label encoding to the 'loan_status' column
df_ml['loan_status'] = label_encoder.fit_transform(df_ml['loan_status'])
# print(df_ml.head())

# Display the updated DataFrame with encoded columns
# print(df_ml[['education', 'self_employed','loan_status']])



# Create a StandardScaler instance
scaler = StandardScaler()

# Define the feature columns (X) and target column (y)
x = df_ml.drop(columns=['loan_status'])  # Drop 'loan_status' column to get feature columns
y = df_ml['loan_status']  # Target variable

# Select only the numerical columns for scaling (excluding 'loan_status')
numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

# Apply scaling to the numerical columns
x[numerical_columns] = scaler.fit_transform(x[numerical_columns])
print(y.head())

# Display the scaled feature variables (X) and the target variable (y)
# print("Scaled Feature Variables (x):")
# print(x.head())

# print("\nTarget Variable (y):")
# print(y.head())
# ------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_result = decision_tree.fit(x_train, y_train)
y_pred_dt = decision_tree.predict(x_test)
# print(y_pred_dt)

print("New Input")
# print(y_test)
# row  = x.iloc[308]
# print(df_ml.iloc[308])
# print(row)

accuracy = accuracy_score(y_test, y_pred_dt)
print("Accuracy:", accuracy)




# Initialize the Decision Tree classifier
# print(df_ml.at[0,'cibil_score'])


# Random Forest

random_forest = RandomForestClassifier(random_state=42)

# Train the random forest model
random_forest.fit(x_train, y_train)

# Predict on the test set
y_pred_rf = random_forest.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy)




#XG Booster

xgb_classifier = XGBClassifier(random_state=42)

# Train the XGBoost model
xgb_classifier.fit(x_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_xgb)
print("Accuracy:", accuracy)




#lightgbm

# Create a LightGBM classifier instance
lgb_classifier = lgb.LGBMClassifier(random_state=42)

# Train the LightGBM model
lgb_classifier.fit(x_train, y_train)

# Predict on the test set
y_pred_lgb = lgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_lgb)
print("Accuracy:", accuracy)




# pickle.dump(decision_tree,open('new_model.pkl','wb'))


# Save the model
filename = 'savedmodel.sav'
pickle.dump(decision_tree,open(filename,'wb'))