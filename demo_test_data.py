import joblib
import pandas as pd

def main():
    # Load the trained model
    model = joblib.load('heart_disease_model.joblib')

    # Create sample test data with two instances to demonstrate both sides of the outcome
    test_data = pd.DataFrame([
        {
            'age': 58,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 1,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 0,
            'thal': 2
        },
        {
            'age': 45,
            'sex': 0,
            'cp': 1,
            'trestbps': 120,
            'chol': 180,
            'fbs': 0,
            'restecg': 0,
            'thalach': 170,
            'exang': 0,
            'oldpeak': 0.0,
            'slope': 1,
            'ca': 0,
            'thal': 3
        }
    ])

    # Predict using the model
    predictions = model.predict(test_data)

    # Print the prediction results
    for i, prediction in enumerate(predictions):
        print(f'Prediction for test data {i+1}: {prediction} (0 = no heart disease, 1 = heart disease)')

if __name__ == '__main__':
    main()
