import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Load dataset
    data = pd.read_csv('heart.csv')

    # Check for missing values
    if data.isnull().sum().sum() > 0:
        data = data.dropna()

    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'heart_disease_model.joblib')
    print('Model saved to heart_disease_model.joblib')

if __name__ == '__main__':
    main()
