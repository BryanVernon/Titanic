# Ensure that the columns in the test dataset match the columns used for training
# X_test = df2[X_train.columns]

# # Initialize the RandomForestClassifier
# clf = RandomForestClassifier(random_state=42)

# # Train the model
# clf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Assuming 'y_pred' contains your predictions and 'PassengerId' is a column in df2
# output_df = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': y_pred})

# # Save the DataFrame to a CSV file
# output_df.to_csv('predictions.csv', index=False)