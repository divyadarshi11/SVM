# SVM
Task 1: SVM with 75%/25% Train/Test Split and Linear Kernel
Training set accuracy: 0.60
Test set accuracy: 0.62

Task 2: Confusion Matrix:

     PF   PG   C   SG   SF   All
PF  12    3     5    4      2    26
PG  1    23    0     0     0    24
C     1    0    20    0     2    23
SG   2   7     0    18    4    31
SF  11   2      1     3     4   21
All  27  35   26    25  12   125




Task 3: 10-fold Cross-validation Accuracy

Fold 1 	 Accuracy: 0.58
Fold 2 	 Accuracy: 0.42
Fold 3 	 Accuracy: 0.54
Fold 4 	 Accuracy: 0.50
Fold 5 	 Accuracy: 0.48
Fold 6 	 Accuracy: 0.52
Fold 7 	 Accuracy: 0.50
Fold 8 	 Accuracy: 0.63
Fold 9 	 Accuracy: 0.55
Fold 10 	 Accuracy: 0.55
Average cross-validation accuracy: 0.53


TASK-4
To achieve better accuracy on the NBA dataset, I followed a series of steps and made some key decisions:
•	Data Preprocessing: The first step was data preprocessing. I loaded the dataset using pandas and identified the feature columns and the target class column. The feature columns represent various statistics related to NBA players, and the target class column (‘Pos’) represents the player positions.

•	Feature Scaling: After loading the data, I scaled the feature columns using StandardScaler from scikit-learn. Scaling the features ensures that all features contribute equally to the model and prevents any feature from dominating the learning process. Scaling is essential for support vector machines as they are sensitive to feature scales.

•	Train/Test Split: I split the data into training and testing sets using the train_test_split function from scikit-learn. I used a 75%/25% split, where 75% of the data was used to train the SVM model, and 25% was used to test its performance.

•	SVM Model with Linear Kernel: I chose the support vector machine (SVM) algorithm with a linear kernel for this classification task. The linear kernel is appropriate when the data can be well separated by a hyperplane in the feature space. SVM is known for its ability to handle high-dimensional data.

•	Hyperparameter Tuning: I set the regularization parameter ‘C’ to 5 and the gamma parameter to ‘Scale’. The C parameter controls the regularization strength in SVM. The gamma parameter controls the kernel coefficient.

•	Model Training: I trained the SVM model on the training data using the fit method.

•	Model Evaluation: After training the model, I evaluated its performance on both the training and testing sets using the accuracy score. The accuracy score calculates the percentage of correct predictions over the total number of predictions.

•	Confusion Matrix: To gain more insights into the model's performance, I generated a confusion matrix on the test set. The confusion matrix helps visualize the number of true positive, true negative, false positive, and false negative predictions for each class.

•	Cross-Validation: To validate the model's performance more rigorously and to assess its generalization ability, I performed 10-fold cross-validation using the cross_val_score function from scikit-learn. 

•	Throughout the process, I iterated and fine-tuned the model and its hyperparameters based on the evaluation results. By monitoring the accuracy scores and confusion matrix, I adjusted to improve the model's overall performance.

REFERENCE:
https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/
https://scikit-learn.org/stable/modules/svm.html
