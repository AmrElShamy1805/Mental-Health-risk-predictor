import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load processed data and encoders
df = pd.read_csv('preprocessed_survey.csv')
label_encoders = joblib.load('label_encoders.pkl')

# Split features and target
X = df.drop(columns=['treatment'])
y = df['treatment']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}
trained_models = {}

# Train Logistic Regression and SVM normally
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_test = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    # Accuracy scores
    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    results[name] = test_acc
    trained_models[name] = model

    # Print metrics
    print(f"\n{name}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(classification_report(y_test, y_pred_test))

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Grid Search for Random Forest
print("\nRunning Grid Search for Random Forest...\n")
param_grid = {
    'n_estimators': [61, 115, 150],
    'max_depth': [None, 4, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model evaluation
best_rf = grid_search.best_estimator_
y_pred_test = best_rf.predict(X_test_scaled)
y_pred_train = best_rf.predict(X_train_scaled)

# Accuracy scores
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
results['Random Forest'] = test_acc
trained_models['Random Forest'] = best_rf

print("\nRandom Forest with Grid Search")
print("Best Parameters:", grid_search.best_params_)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(classification_report(y_test, y_pred_test))

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_random_forest.png')
plt.close()

# Model accuracy comparison plot
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Save best model overall
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
joblib.dump(trained_models[best_model_name], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
