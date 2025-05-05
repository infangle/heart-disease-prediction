# heart-disease-prediction

## Project Description
This project is a machine learning-based heart disease prediction system. It uses a dataset of patient health metrics to train a model that can predict the likelihood of heart disease. The project includes scripts for training the model and testing it with sample data.

## Cloning the Repository
To clone this repository, use the following command:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

## Running the Project

### Prerequisites
- Python 3.x installed on your system
- Recommended to use a virtual environment

### Install Dependencies
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

(Note: If `requirements.txt` is not present, you may need to install packages such as `pandas`, `scikit-learn`, and `joblib` manually.)

### Train the Model
To train the heart disease prediction model, run:

```bash
python train_heart_disease_model.py
```

### Test the Model
To test the model with demo data, run:

```bash
python demo_test_data.py
```

This will use the trained model to make predictions on sample data.

## Project Files
- `train_heart_disease_model.py`: Script to train the heart disease prediction model.
- `demo_test_data.py`: Script to test the trained model with sample data.
- `heart_disease_model.joblib`: The saved trained model file.
- `heart.csv`: Dataset used for training the model.

## License
This project is licensed under the MIT License.
