import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

def train_model():
    # Load data
    data = pd.read_csv('heart_disease_uci.csv')
    data.dropna(inplace = True)

    # Shuffle data
    data = shuffle(data)

    # Data preprocessing
    data['thal'].replace({'fixed defect':'fixed_defect', 'reversable defect': 'reversable_defect'}, inplace=True)
    data['cp'].replace({'typical angina':'typical_angina', 'atypical angina': 'atypical_angina'}, inplace=True)

    # Prepare data for modeling
    data_tmp = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
    data_tmp['target'] = ((data['num'] > 0) * 1).copy()
    data_tmp['sex'] = (data['sex'] == 'Male') * 1
    data_tmp['fbs'] = (data['fbs']) * 1
    data_tmp['exang'] = (data['exang']) * 1

    # Renaming columns
    data_tmp.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                        'cholesterol', 'fasting_blood_sugar',
                        'max_heart_rate_achieved', 'exercise_induced_angina', 
                        'st_depression', 'st_slope_type', 'num_major_vessels', 
                        'thalassemia_type', 'target']

    # One-hot encoding
    data = pd.get_dummies(data_tmp, drop_first=False)

    # Split data into features and target
    y = data['target']
    X = data.drop('target', axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train)).values
    X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)).values

    # Initialize and train the logistic regression model
    logre = LogisticRegression()
    logre.fit(X_train, y_train)

    # Making predictions
    y_pred = logre.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'The Accuracy Score is: {accuracy}')

    # Return the trained model
    return logre

# Train the model
logre = train_model()

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_data = request.form

    # Retrieve and preprocess form data to match model input format
    # Make sure to convert string inputs to float or int as necessary
    age = float(form_data['age'])
    sex = 1 if form_data['sex'].lower() == 'male' else 0
    chest_pain_type = form_data['chest_pain_type']
    resting_blood_pressure = float(form_data['resting_blood_pressure'])
    cholesterol = float(form_data['cholesterol'])
    fasting_blood_sugar = float(form_data['fasting_blood_sugar'])
    max_heart_rate_achieved = float(form_data['max_heart_rate_achieved'])
    exercise_induced_angina = float(form_data['exercise_induced_angina'])
    st_depression = float(form_data['st_depression'])
    st_slope_type = form_data['st_slope_type']
    num_major_vessels = int(form_data['num_major_vessels'])
    thalassemia_type = form_data['thalassemia_type']

    # Convert categorical variables to one-hot encoded format as expected by the model
    input_data = [
        age, 
        sex, 
        resting_blood_pressure, 
        cholesterol, 
        fasting_blood_sugar, 
        max_heart_rate_achieved, 
        exercise_induced_angina, 
        st_depression, 
        num_major_vessels,
        1 if chest_pain_type == 'typical_angina' else 0,
        1 if chest_pain_type == 'atypical_angina' else 0,
        1 if chest_pain_type == 'non_anginal_pain' else 0,
        1 if chest_pain_type == 'asymptomatic' else 0,
        1 if st_slope_type == 'upsloping' else 0,
        1 if st_slope_type == 'flat' else 0,
        1 if st_slope_type == 'downsloping' else 0,
        1 if thalassemia_type == 'fixed_defect' else 0,
        1 if thalassemia_type == 'normal' else 0,
        1 if thalassemia_type == 'reversable_defect' else 0
    ]
    input_data = np.array(input_data).reshape(1, -1)

    # Make a prediction
    prediction = logre.predict(input_data)
    print(prediction)
    # Convert prediction 
    prediction_message = "Positive for heart disease" if prediction[0] == 1 else "Negative for heart disease"

    # Redirect to prediction.html with the prediction result
    return render_template('prediction.html', prediction_result=prediction_message)

if __name__ == '__main__':
    app.run(debug=True,port=5002)