import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)  # Displays all columns
pd.set_option('display.expand_frame_repr', False)  # Disables line wrapping to display data horizontally

# Load data
df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

# EDA
# sns.heatmap(df.isnull(), cmap='viridis') # Visualize null values
# plt.show()
# sns.countplot(x='Survived', hue='Sex', data=df, palette='RdBu_r') # Visualize the relationship between sex and survival
# plt.show()
# sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Pclass', data=df, palette='rainbow') # Visualize the relationship between class and survival
# plt.show()

# Data Wrangling

df = df.drop('Cabin', axis=1)
# Convert 'Sex' column to numeric values
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Convert 'Embarked' column to numeric values
le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Separate dataset into two parts: one with missing ages and one without
df_missing_age = df[df['Age'].isnull()]
df_no_missing_age = df[df['Age'].notnull()]

# Features and target variable
features = ['Pclass', 'SibSp', 'Parch', 'Fare','Embarked','Sex']
target = 'Age'

# Train a model to predict missing ages
model = RandomForestRegressor()
model.fit(df_no_missing_age[features], df_no_missing_age[target])

# Predict missing ages using .loc
df.loc[df['Age'].isnull(), 'Age'] = model.predict(df_missing_age[features])

# Handle missing values in the 'Embarked' column
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Add a Title column
df['Title'] = df['Name'].str.extract(r',\s(.*?)\.')

# Drop unnecessary columns
df = df.drop(columns= ['Name', 'Ticket'])

# print(df['Title'].unique())

df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Rare')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)

# print(df.isna().sum())

# Data Wrangling for test dataset

df2 = df2.drop('Cabin', axis=1)
df2['Fare'].fillna(df2['Fare'].median(), inplace=True)
# Convert 'Sex' column to numeric values
le_sex = LabelEncoder()
df2['Sex'] = le_sex.fit_transform(df2['Sex'])

# Convert 'Embarked' column to numeric values
le_embarked = LabelEncoder()
df2['Embarked'] = le_embarked.fit_transform(df2['Embarked'])

# Separate dataset into two parts: one with missing ages and one without
df_missing_age = df2[df2['Age'].isnull()]
df_no_missing_age = df2[df2['Age'].notnull()]



# Features and target variable
features = ['Pclass', 'SibSp', 'Parch', 'Fare','Embarked','Sex']
target = 'Age'

# Train a model to predict missing ages
model = RandomForestRegressor()
model.fit(df_no_missing_age[features], df_no_missing_age[target])

# Predict missing ages using .loc
df2.loc[df2['Age'].isnull(), 'Age'] = model.predict(df_missing_age[features])

# Handle missing values in the 'Embarked' column
df2['Embarked'].fillna(df2['Embarked'].mode()[0], inplace=True)

# Add a Title column
df2['Title'] = df2['Name'].str.extract(r',\s(.*?)\.')

# Drop unnecessary columns
df2 = df2.drop(columns= ['Name', 'Ticket'])

# print(df2['Title'].unique())

df2['Title'] = df2['Title'].replace('Mlle', 'Miss')
df2['Title'] = df2['Title'].replace('Ms', 'Miss')
df2['Title'] = df2['Title'].replace('Mme', 'Mrs')
df2['Title'] = df2['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Rare')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df2['Title'] = df2['Title'].map(title_mapping)

# print(df.shape)
# print(df2.shape)
# print(df.head())
# print(df2.head())

# print(df2.isna().sum())

# Keep only the specified features
selected_features = ['Title', 'Pclass', 'Sex', 'Fare', 'SibSp', 'Age', 'Parch', 'Embarked', 'PassengerId']
df = df[selected_features + ['Survived']]
df2 = df2[selected_features]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(df2)

# Assuming 'y_pred' contains your predictions and 'PassengerId' is a column in df2
output_df = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': y_pred})

# Save the DataFrame to a CSV file
output_df.to_csv('pred.csv', index=False)


