import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import mlflow
import matplotlib.pyplot as plt
import dagshub
dagshub.init(repo_owner='ShashankraiOO7', repo_name='mlfow_dagshub_experiment_tracking', mlflow=True)

mlflow.set_tracking_uri('https://github.com/ShashankraiOO7/mlfow_dagshub_experiment_tracking.git')

# Load the dataset
df = pd.read_csv(r'./data/classification_data_5000_records.csv')

# Split the dataset into features and target variable
X = df.drop("output", axis=1)
y = df["output"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize hyperparameters
n_estimators = 100
max_depth = 7

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Set MLflow experiment
mlflow.set_experiment('new one')

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=y.unique(), yticklabels=y.unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Save and log the confusion matrix
plt.savefig("confusion_matrix.png")
plt.close()

# Log metrics, parameters, and artifacts with MLflow
with mlflow.start_run(run_name='444'):
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision_score(y_test, y_pred, average='binary'))
    mlflow.log_metric('recall', recall_score(y_test, y_pred, average='binary'))
    mlflow.log_metric('f1-score', f1_score(y_test, y_pred, average='binary'))
    
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_artifact("confusion_matrix.png")
