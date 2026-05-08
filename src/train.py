import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load dataset
df = pd.read_csv('data/cleaned_heart.csv')

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# MLflow experiment
with mlflow.start_run():

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    preds = pipeline.predict(X_test)

    probs = pipeline.predict_proba(X_test)[:,1]

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    # Log params
    mlflow.log_param('model', 'RandomForest')
    mlflow.log_param('n_estimators', 200)
    mlflow.log_param('max_depth', 10)

    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('roc_auc', roc_auc)

    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path='model'
    )

    # Save locally
    joblib.dump(
        pipeline,
        'models/model.pkl'
    )

print('Training completed successfully')