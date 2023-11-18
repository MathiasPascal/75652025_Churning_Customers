from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load the trained Keras model
model = load_model('model.h5')  # Make sure to replace with the correct file path
scaler = StandardScaler()
final_data = pd.read_csv('final_data.csv')


# HTML form route
@app.route('/')
def churn_prediction_form():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_churn():
    # Extracting user input from the form
    total_charges = float(request.form['totalCharges'])
    monthly_charges = float(request.form['monthlyCharges'])
    tenure = float(request.form['tenure'])
    contract = request.form['contract']
    payment_method = request.form['paymentMethod']
    scaler.fit(final_data.drop('Churn', axis = 1))
    # Mapping categorical values to numerical values based on your preprocessing
    # You might need to adjust this based on how your original preprocessing was done
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_method_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

    # Transforming categorical values to numerical values
    contract_numeric = contract_mapping.get(contract, 0)
    payment_method_numeric = payment_method_mapping.get(payment_method, 0)

    # Creating a numpy array for model prediction
    input_data = np.array([[total_charges, monthly_charges, tenure, contract_numeric, payment_method_numeric]])
    scaled_input_data = scaler.transform(input_data)

    # Making the prediction
    prediction = model.predict({'numerical_input': scaled_input_data[:, :-2], 'categorical_input': scaled_input_data[:, -2:]})
    churn_probability = prediction[0, 0]

    # Determining churn status based on a threshold (e.g., 0.5)
    churn_status = 'Churn' if churn_probability >= 0.5 else 'No Churn'

    # Displaying the prediction result on a new HTML page
    return render_template('result.html', prediction=churn_status, confidence=churn_probability * 100)

if __name__ == '__main__':
    app.run(debug=True)
