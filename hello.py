import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('best.pkl', 'rb') as f:
    model = pickle.load(f)

target = ['No Heart Disease', 'Heart Disease']

# Define a function to make predictions
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a dictionary with the input features
    input_dict = {'age': age,
                  'sex': sex,
                  'cp': cp,
                  'trestbps': trestbps,
                  'chol': chol,
                  'fbs': fbs,
                  'restecg': restecg,
                  'thalach': thalach,
                  'exang': exang,
                  'oldpeak': oldpeak,
                  'slope': slope,
                  'ca': ca,
                  'thal': thal}
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_dict])
    # Scale the input features using the same scaler used during training
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)
    # Make a prediction using the trained model
    prediction = model.predict(input_scaled)
    # Return the predicted class
    return target[int(prediction)]

# Create a web app using Streamlit
st.title('Heart Disease Prediction')
st.write('Please enter the following information to predict the likelihood of having heart disease.')

# Create input fields for user to enter data
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
chol = st.slider('Serum Cholesterol (mg/dL)', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.slider('Maximum Heart Rate Achieved', 50, 220, 150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 0.0, 0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# Convert sex input to binary
if sex == 'Male':
    sex = 1
else:
    sex = 0

# Add a button to make a prediction
if st.button('Predict'):
    # Call the prediction function
    result = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # Display the predicted result
    if 'result' in locals():
        st.write('The predicted likelihood of having heart disease is:', result)
