import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
nba_dataset = pd.read_csv('nba2021.csv')
original_headers = list(nba_dataset.columns.values)

# Identify the target class and feature columns
class_column = 'Pos'
feature_columns = ['FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2P%', 'FT%', 'eFG%', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']
nba_feature = nba_dataset[feature_columns]
nba_class = nba_dataset[class_column]

# Scale the features using StandardScaler
scaler = StandardScaler()
nba_feature_scaled = pd.DataFrame(scaler.fit_transform(nba_feature))

# Task 1: Train/Test Split SVM Model with Linear Kernel
train_feature, test_feature, train_class, test_class = train_test_split(
    nba_feature_scaled, nba_class, stratify=nba_class, train_size=0.75, test_size=0.25, random_state=1)

# Build and train the SVM model with a linear kernel
linearsvm = SVC(kernel='linear', C=5, gamma='scale', random_state=1)
linearsvm.fit(train_feature, train_class)

# Evaluate the model on the test set
train_accuracy = linearsvm.score(train_feature, train_class)
test_accuracy = linearsvm.score(test_feature, test_class)

# Print the results of Task 1
print("Task 1: SVM with 75%/25% Train/Test Split and Linear Kernel\n")
print("Training set accuracy: {:.2f}".format(train_accuracy))
print("Test set accuracy: {:.2f}".format(test_accuracy))

# Task 2: Print Confusion Matrix
test_predictions = linearsvm.predict(test_feature)

# Compute the confusion matrix with the correct labels for rows and columns
confusion_mat = confusion_matrix(test_class, test_predictions, labels=nba_dataset[class_column].unique())
confusion_df = pd.DataFrame(confusion_mat, index=nba_dataset[class_column].unique(), columns=nba_dataset[class_column].unique())
confusion_df['All'] = confusion_df.sum(axis=1)
confusion_df.loc['All'] = confusion_df.sum(axis=0)

# Print the confusion matrix
print("\nTask 2: Confusion Matrix:\n")
print(confusion_df)

# Task 3: 10-fold Stratified Cross-validation
linearsvm_crossval = SVC(kernel='linear', C=5, gamma='scale', random_state=1)

# Perform 10-fold cross-validation and get accuracy scores for each fold
crossval_accuracy = cross_val_score(linearsvm_crossval, nba_feature_scaled, nba_class, cv=10, scoring='accuracy')

# Print accuracy for each fold
print("\nTask 3: 10-fold Cross-validation Accuracy\n")
for i, accuracy in enumerate(crossval_accuracy, 1):
    print("Fold {} \t Accuracy: {:.2f}".format(i, accuracy))

# Print average accuracy across all folds
print("\nAverage cross-validation accuracy: {:.2f}".format(np.mean(crossval_accuracy)))

