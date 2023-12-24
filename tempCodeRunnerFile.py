# Predict missing ages using .loc
df2.loc[df2['Age'].isnull(), 'Age'] = model.predict(df_missing_age[features])

# Handle missing values in the 'Embarked' column
df2['Embarked'].fillna(df2['Embarked'].mode()[0], inplace=True)

# Add a Title column
df2['Title'] = df2['Name'].str.extract(r',\s(.*?)\.')

# Drop unnecessary columns
df2 = df2.drop(columns= ['Name','PassengerId', 'Ticket'])

# Convert 'Sex' column to numeric values
le_sex = LabelEncoder()
df2['Sex'] = le_sex.fit_transform(df2['Sex'])

# Convert 'Embarked' column to numeric values
le_embarked = LabelEncoder()
df2['Embarked'] = le_embarked.fit_transform(df2['Embarked'])

print(df2['Title'].unique())

df2['Title'] = df2['Title'].replace('Mlle', 'Miss')
df2['Title'] = df2['Title'].replace('Ms', 'Miss')
df2['Title'] = df2['Title'].replace('Mme', 'Mrs')
df2['Title'] = df2['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Rare')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df2['Title'] = df2['Title'].map(title_mapping)

print(df.shape)
print(df2.shape)
print(df.head())
print(df2.head())

print(df2.isna().sum())


